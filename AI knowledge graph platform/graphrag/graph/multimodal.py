"""Multi-modal entity attachments — images, audio, and video linked to KG entities.

Problem solved
--------------
The current graph is text-only.  Real-world entities (persons, products,
locations) have associated images, logos, satellite photos, product photos,
audio transcripts, video clips, etc.  Without multi-modal support:
  - Product search cannot surface visual variants.
  - Person entity resolution ignores facial recognition signals.
  - Document chunks containing image captions lack entity links.

Architecture
------------
MediaAttachment nodes are stored in Neo4j and linked to Entity nodes via
HAS_MEDIA edges.  The actual media bytes are NOT stored in the graph —
only references (URL or object-store key) plus extracted metadata:

  Entity -[:HAS_MEDIA]-> MediaAttachment {
      id, entity_name, entity_type, tenant,
      modality,     # image | audio | video | document
      media_url,    # s3://... or https://... or local path
      caption,      # human-readable or auto-generated description
      embedding,    # optional CLIP/audio embedding for cross-modal search
      mime_type,
      created_at,
  }

Cross-modal retrieval: if a MediaAttachment has an embedding, it participates
in the ANN vector search stage alongside text embeddings.

Note: this module is a Phase 1 implementation of multi-modal support.
Actual embedding computation (CLIP for images, Whisper for audio) is
intentionally left as a hook — call `set_embedding()` after computing the
embedding externally.
"""

from __future__ import annotations

from uuid import uuid4

import structlog

log = structlog.get_logger(__name__)

_VALID_MODALITIES = frozenset({"image", "audio", "video", "document"})


class MultiModalEntityService:
    """
    Attach and retrieve multi-modal media references for KG entities.

    Usage::

        svc = MultiModalEntityService(neo4j_client)

        # Attach a product image
        attachment_id = await svc.attach_media(
            entity_name="Boeing 737",
            entity_type="PRODUCT",
            tenant="acme",
            modality="image",
            media_url="s3://acme-media/737.jpg",
            caption="Boeing 737 MAX in British Airways livery",
            mime_type="image/jpeg",
        )

        # Attach an audio transcript excerpt
        await svc.attach_media(
            entity_name="Elon Musk",
            entity_type="PERSON",
            tenant="acme",
            modality="audio",
            media_url="s3://acme-media/musk_interview.mp3",
            caption="Interview excerpt discussing SpaceX Mars plans",
        )

        # Store a cross-modal embedding after computing it externally
        await svc.set_embedding(attachment_id, clip_embedding_vector)

        # Retrieve all media for an entity
        media = await svc.get_modalities("Boeing 737", "PRODUCT", "acme")
    """

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    # ── Attach ─────────────────────────────────────────────────────────────────

    async def attach_media(
        self,
        entity_name: str,
        entity_type: str,
        tenant: str = "default",
        modality: str = "image",
        media_url: str = "",
        caption: str = "",
        mime_type: str = "",
    ) -> str:
        """
        Attach a media reference to an entity.

        Creates a MediaAttachment node and a HAS_MEDIA edge from the entity.
        If the entity does not yet exist in the graph, the HAS_MEDIA edge
        creation is skipped silently (MediaAttachment is still created for
        future linking).

        Returns the attachment id.

        Parameters
        ----------
        modality  : One of: image | audio | video | document
        media_url : URL or object-store key pointing to the actual media bytes.
        caption   : Human-readable description (used as text for ANN search).
        mime_type : Optional MIME type for storage/display hints.
        """
        if modality not in _VALID_MODALITIES:
            raise ValueError(
                f"Unknown modality {modality!r}. Must be one of: {sorted(_VALID_MODALITIES)}"
            )

        attachment_id = str(uuid4())
        await self._neo4j.run(
            """
            CREATE (m:MediaAttachment {
                id:           $id,
                entity_name:  $entity_name,
                entity_type:  $entity_type,
                tenant:       $tenant,
                modality:     $modality,
                media_url:    $media_url,
                caption:      $caption,
                mime_type:    $mime_type,
                created_at:   datetime()
            })
            WITH m
            OPTIONAL MATCH (e:Entity {name: $entity_name, type: $entity_type, tenant: $tenant})
            FOREACH (x IN CASE WHEN e IS NOT NULL THEN [1] ELSE [] END |
                MERGE (e)-[:HAS_MEDIA]->(m)
            )
            """,
            id=attachment_id,
            entity_name=entity_name,
            entity_type=entity_type,
            tenant=tenant,
            modality=modality,
            media_url=media_url,
            caption=caption,
            mime_type=mime_type,
        )
        log.info(
            "multimodal.attached",
            attachment_id=attachment_id,
            entity=entity_name,
            modality=modality,
            tenant=tenant,
        )
        return attachment_id

    async def attach_image(
        self,
        entity_name: str,
        entity_type: str,
        tenant: str = "default",
        image_url: str = "",
        caption: str = "",
        mime_type: str = "image/jpeg",
    ) -> str:
        """Convenience wrapper — attach an image attachment."""
        return await self.attach_media(
            entity_name=entity_name,
            entity_type=entity_type,
            tenant=tenant,
            modality="image",
            media_url=image_url,
            caption=caption,
            mime_type=mime_type,
        )

    async def attach_audio(
        self,
        entity_name: str,
        entity_type: str,
        tenant: str = "default",
        audio_url: str = "",
        transcript: str = "",
        mime_type: str = "audio/mpeg",
    ) -> str:
        """
        Convenience wrapper — attach an audio file with its transcript as caption.
        """
        return await self.attach_media(
            entity_name=entity_name,
            entity_type=entity_type,
            tenant=tenant,
            modality="audio",
            media_url=audio_url,
            caption=transcript,
            mime_type=mime_type,
        )

    # ── Retrieve ───────────────────────────────────────────────────────────────

    async def get_modalities(
        self,
        entity_name: str,
        entity_type: str,
        tenant: str = "default",
    ) -> list[dict]:
        """Return all MediaAttachment records for an entity."""
        rows = await self._neo4j.run(
            """
            MATCH (m:MediaAttachment {
                entity_name: $entity_name,
                entity_type: $entity_type,
                tenant:      $tenant
            })
            RETURN m.id          AS id,
                   m.modality    AS modality,
                   m.media_url   AS media_url,
                   m.caption     AS caption,
                   m.mime_type   AS mime_type,
                   m.created_at  AS created_at,
                   (m.embedding IS NOT NULL) AS has_embedding
            ORDER BY m.created_at DESC
            """,
            entity_name=entity_name,
            entity_type=entity_type,
            tenant=tenant,
        )
        return [dict(r) for r in rows]

    async def list_by_modality(
        self,
        modality: str,
        tenant: str = "default",
        limit: int = 100,
    ) -> list[dict]:
        """Return all MediaAttachments of a given modality for a tenant."""
        rows = await self._neo4j.run(
            """
            MATCH (m:MediaAttachment {modality: $modality, tenant: $tenant})
            RETURN m.id          AS id,
                   m.entity_name AS entity_name,
                   m.entity_type AS entity_type,
                   m.media_url   AS media_url,
                   m.caption     AS caption,
                   m.created_at  AS created_at
            ORDER BY m.created_at DESC
            LIMIT $limit
            """,
            modality=modality,
            tenant=tenant,
            limit=limit,
        )
        return [dict(r) for r in rows]

    # ── Embedding management ───────────────────────────────────────────────────

    async def set_embedding(
        self,
        attachment_id: str,
        embedding: list[float],
    ) -> None:
        """
        Store a cross-modal embedding on a MediaAttachment.

        Call this after computing the embedding externally (e.g. via CLIP for
        images, or a sentence transformer on the caption text).  Once stored,
        the attachment participates in ANN vector retrieval.
        """
        await self._neo4j.run(
            """
            MATCH (m:MediaAttachment {id: $id})
            SET m.embedding        = $embedding,
                m.embedding_dim    = $dim,
                m.embedding_set_at = datetime()
            """,
            id=attachment_id,
            embedding=embedding,
            dim=len(embedding),
        )
        log.info("multimodal.embedding_stored",
                 attachment_id=attachment_id, dim=len(embedding))

    async def get_unembedded(
        self,
        tenant: str = "default",
        modality: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Return MediaAttachments that have no embedding yet.

        Use this to drive a batch embedding job.
        """
        modality_filter = "AND m.modality = $modality" if modality else ""
        params: dict = {"tenant": tenant, "limit": limit}
        if modality:
            params["modality"] = modality

        rows = await self._neo4j.run(
            f"""
            MATCH (m:MediaAttachment {{tenant: $tenant}})
            WHERE m.embedding IS NULL
            {modality_filter}
            RETURN m.id          AS id,
                   m.entity_name AS entity_name,
                   m.modality    AS modality,
                   m.media_url   AS media_url,
                   m.caption     AS caption
            ORDER BY m.created_at DESC
            LIMIT $limit
            """,
            **params,
        )
        return [dict(r) for r in rows]

    # ── Deletion ───────────────────────────────────────────────────────────────

    async def delete_attachment(self, attachment_id: str) -> bool:
        """Remove a MediaAttachment and its HAS_MEDIA edge."""
        rows = await self._neo4j.run(
            """
            MATCH (m:MediaAttachment {id: $id})
            DETACH DELETE m
            RETURN count(m) AS n
            """,
            id=attachment_id,
        )
        deleted = rows[0].get("n", 0) if rows else 0
        return bool(deleted)

    async def delete_entity_media(
        self,
        entity_name: str,
        entity_type: str,
        tenant: str = "default",
    ) -> int:
        """Remove ALL MediaAttachments for an entity (e.g. on GDPR erasure)."""
        rows = await self._neo4j.run(
            """
            MATCH (m:MediaAttachment {
                entity_name: $entity_name,
                entity_type: $entity_type,
                tenant:      $tenant
            })
            DETACH DELETE m
            RETURN count(m) AS n
            """,
            entity_name=entity_name,
            entity_type=entity_type,
            tenant=tenant,
        )
        count = rows[0].get("n", 0) if rows else 0
        if count:
            log.info("multimodal.entity_media_deleted",
                     entity=entity_name, count=count)
        return count
