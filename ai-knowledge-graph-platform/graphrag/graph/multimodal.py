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
from pydantic import BaseModel, Field

import structlog

from graphrag.core.tenancy import require_tenant

log = structlog.get_logger(__name__)

_VALID_MODALITIES = frozenset({"image", "audio", "video", "document"})


class MediaTransformation(BaseModel):
    """A provenance-preserving derived representation of a media artifact."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    tenant: str = "default"
    input_attachment_id: str
    output_artifact_id: str
    transform_type: str  # ocr | transcript | visual_embedding | caption
    model_version: str = ""
    output_digest: str = ""
    metadata: dict = Field(default_factory=dict)


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
        await svc.set_embedding("acme", attachment_id, clip_embedding_vector)

        # Retrieve all media for an entity
        media = await svc.get_modalities("Boeing 737", "PRODUCT", "acme")
    """

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    async def record_transformation(self, transformation: MediaTransformation) -> str:
        """Store OCR/transcript/embedding provenance without copying media bytes."""
        await self._neo4j.run(
            """
            MATCH (m:MediaAttachment {id: $input_id, tenant: $tenant})
            MERGE (a:SourceArtifact {id: $output_id, tenant: $tenant})
            SET a.modality = m.modality, a.transform_type = $transform_type,
                a.model_version = $model_version, a.output_digest = $output_digest,
                a.metadata = $metadata, a.created_at = datetime()
            MERGE (m)-[:TRANSFORMED_TO {type: $transform_type}]->(a)
            """,
            input_id=transformation.input_attachment_id,
            output_id=transformation.output_artifact_id,
            tenant=transformation.tenant,
            transform_type=transformation.transform_type,
            model_version=transformation.model_version,
            output_digest=transformation.output_digest,
            metadata=transformation.metadata,
        )
        return transformation.id

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
        image_bytes: bytes | None = None,
    ) -> str:
        """Convenience wrapper — attach an image attachment.

        If ``image_bytes`` is given, a perceptual hash is computed inline
        (via ``graphrag.graph.perceptual_hash.compute_phash``) and stored on
        the same MediaAttachment node, so ``find_similar_images`` can use it
        immediately without a separate ``set_perceptual_hash`` call.
        """
        attachment_id = await self.attach_media(
            entity_name=entity_name,
            entity_type=entity_type,
            tenant=tenant,
            modality="image",
            media_url=image_url,
            caption=caption,
            mime_type=mime_type,
        )
        if image_bytes is not None:
            from graphrag.graph.perceptual_hash import compute_phash
            phash = compute_phash(image_bytes)
            await self.set_perceptual_hash(tenant, attachment_id, phash)
        return attachment_id

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
        tenant: str,
        attachment_id: str,
        embedding: list[float],
    ) -> None:
        """
        Store a cross-modal embedding on a MediaAttachment.

        Call this after computing the embedding externally (e.g. via CLIP for
        images, or a sentence transformer on the caption text).  Once stored,
        the attachment participates in ANN vector retrieval.

        `tenant` required: attachment_id is a UUID (collision-safe), but
        without a tenant check here a caller from any tenant who learns
        another tenant's attachment_id (log line, error message, timing)
        could silently overwrite that attachment's embedding.
        """
        tenant = require_tenant(tenant)
        await self._neo4j.run(
            """
            MATCH (m:MediaAttachment {id: $id, tenant: $tenant})
            SET m.embedding        = $embedding,
                m.embedding_dim    = $dim,
                m.embedding_set_at = datetime()
            """,
            id=attachment_id,
            tenant=tenant,
            embedding=embedding,
            dim=len(embedding),
        )
        log.info("multimodal.embedding_stored",
                 attachment_id=attachment_id, dim=len(embedding), tenant=tenant)

    # ── Perceptual hashing ─────────────────────────────────────────────────────

    async def set_perceptual_hash(
        self,
        tenant: str,
        attachment_id: str,
        phash: str,
    ) -> None:
        """Store a perceptual hash (see ``graphrag.graph.perceptual_hash``) on
        a MediaAttachment, and record the computation as a provenance-tracked
        transformation.

        `tenant` required for the same reason as ``set_embedding``: without a
        tenant check, a caller who learns another tenant's attachment_id
        could silently overwrite that attachment's hash.
        """
        tenant = require_tenant(tenant)
        await self._neo4j.run(
            """
            MATCH (m:MediaAttachment {id: $id, tenant: $tenant})
            SET m.phash = $phash, m.phash_set_at = datetime()
            """,
            id=attachment_id,
            tenant=tenant,
            phash=phash,
        )
        await self.record_transformation(MediaTransformation(
            tenant=tenant,
            input_attachment_id=attachment_id,
            output_artifact_id=attachment_id,
            transform_type="perceptual_hash",
            output_digest=phash,
        ))
        log.info("multimodal.phash_stored", attachment_id=attachment_id, tenant=tenant)

    async def find_similar_images(
        self,
        tenant: str,
        attachment_id: str,
        max_distance: int = 8,
        limit: int = 20,
    ) -> list[dict]:
        """Find image MediaAttachments perceptually similar to a given one.

        Fetches the tenant's image attachments that have a stored phash and
        computes Hamming distance against ``attachment_id``'s hash in Python
        (Neo4j has no native Hamming-distance operator, and tenant-scoped
        image counts don't warrant a dedicated similarity index). Returns
        matches within ``max_distance``, closest first. The target itself is
        excluded from results.

        Returns an empty list if ``attachment_id`` has no stored phash.
        """
        from graphrag.graph.perceptual_hash import hamming_distance

        tenant = require_tenant(tenant)
        rows = await self._neo4j.run(
            """
            MATCH (m:MediaAttachment {modality: 'image', tenant: $tenant})
            WHERE m.phash IS NOT NULL
            RETURN m.id AS id, m.entity_name AS entity_name,
                   m.entity_type AS entity_type, m.media_url AS media_url,
                   m.caption AS caption, m.phash AS phash
            """,
            tenant=tenant,
        )
        rows = [dict(r) for r in rows]
        target = next((r for r in rows if r["id"] == attachment_id), None)
        if target is None:
            return []

        matches = []
        for r in rows:
            if r["id"] == attachment_id:
                continue
            distance = hamming_distance(target["phash"], r["phash"])
            if distance <= max_distance:
                matches.append({**{k: v for k, v in r.items() if k != "phash"},
                                 "distance": distance})
        matches.sort(key=lambda r: r["distance"])
        return matches[:limit]

    # ── OCR ────────────────────────────────────────────────────────────────────

    async def run_ocr(
        self,
        tenant: str,
        attachment_id: str,
        image_bytes: bytes,
    ) -> MediaTransformation:
        """Run OCR over an image attachment and record the result.

        Stores the extracted text as a provenance-tracked
        ``MediaTransformation`` (transform_type="ocr"), and — only if the
        attachment's caption is currently empty — backfills the caption with
        the OCR'd text, so it becomes searchable through the existing
        caption-based ANN/text retrieval path without any new plumbing.
        """
        from graphrag.graph.ocr import extract_text

        tenant = require_tenant(tenant)
        text, confidence = extract_text(image_bytes)

        transformation = MediaTransformation(
            tenant=tenant,
            input_attachment_id=attachment_id,
            output_artifact_id=attachment_id,
            transform_type="ocr",
            output_digest=text,
            metadata={"confidence": confidence},
        )
        await self.record_transformation(transformation)

        await self._neo4j.run(
            """
            MATCH (m:MediaAttachment {id: $id, tenant: $tenant})
            WHERE m.caption IS NULL OR m.caption = ''
            SET m.caption = $text
            """,
            id=attachment_id,
            tenant=tenant,
            text=text,
        )
        log.info("multimodal.ocr_run", attachment_id=attachment_id,
                 tenant=tenant, chars=len(text), confidence=confidence)
        return transformation

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

    async def delete_attachment(self, tenant: str, attachment_id: str) -> bool:
        """Remove a MediaAttachment and its HAS_MEDIA edge."""
        tenant = require_tenant(tenant)
        rows = await self._neo4j.run(
            """
            MATCH (m:MediaAttachment {id: $id, tenant: $tenant})
            DETACH DELETE m
            RETURN count(m) AS n
            """,
            id=attachment_id,
            tenant=tenant,
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
