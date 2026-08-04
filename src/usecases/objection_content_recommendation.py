"""§12 first wired use case. Given an Opportunity:

1. find the most recent relevant call
2. identify an affirmed, non-rejected Objection raised by a buyer stakeholder
3. identify ContentAssets that address the Objection through curated tags or
   explicit mappings
4. exclude assets already viewed by that buyer
5. rank remaining assets
6. return the recommendation, exact transcript evidence, Claim IDs, and
   ranking explanation

Objection<->ContentAsset mapping source: ContentAsset.tags, a curated field set
by the content team at ingestion time (src/ingestion/adapters/showpad.py) — the
match is `objection_category in {tag.lower() for tag in asset.tags}`, never a
runtime-invented guess (§12: 'It may come from a curated taxonomy or reviewed
Claim; it must not be invented at query time'). config/ontologies/sales.yml's
ADDRESSES_OBJECTION relation_rule documents this domain/target pairing at the
ontology level; this is its concrete data-level implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.domain.assertion import Claim
from src.domain.enums import AdjudicationStatus, Polarity, SpeakerRole
from src.domain.knowledge import ContentAsset
from src.graph.repositories.claim_repository import ClaimRepository
from src.graph.repositories.content_repository import ContentRepository
from src.graph.repositories.conversation_repository import ConversationRepository

MAPPING_SOURCE = "content_asset.tags (curated Showpad content taxonomy)"


class NoRelevantCallError(RuntimeError):
    pass


class NoObjectionFoundError(RuntimeError):
    pass


@dataclass(frozen=True)
class RankedAsset:
    asset: ContentAsset
    matched_tags: list[str]
    rank_score: float


@dataclass(frozen=True)
class ObjectionContentRecommendation:
    opportunity_id: str
    conversation_id: str
    objection_claim: Claim
    evidence_text: str
    recommended_asset: ContentAsset | None
    ranked_candidates: list[RankedAsset]
    excluded_viewed_asset_ids: list[str]
    mapping_source: str
    explanation: str


def _select_objection_claim(claims: list[Claim]) -> Claim | None:
    candidates = [
        c for c in claims
        if c.predicate == "RAISED_OBJECTION"
        and c.polarity == Polarity.AFFIRMED
        and c.adjudication_status != AdjudicationStatus.REJECTED
        and c.speaker_role == SpeakerRole.BUYER
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.source_timestamp)


class ObjectionContentRecommendationUseCase:
    def __init__(
        self,
        conversation_repo: ConversationRepository,
        claim_repo: ClaimRepository,
        content_repo: ContentRepository,
    ):
        self._conversation_repo = conversation_repo
        self._claim_repo = claim_repo
        self._content_repo = content_repo

    async def recommend(
        self, workspace_id: str, opportunity_id: str, buyer_contact_id: str
    ) -> ObjectionContentRecommendation:
        conversations = await self._conversation_repo.list_conversations_by_opportunity(
            workspace_id, opportunity_id
        )
        if not conversations:
            raise NoRelevantCallError(f"no conversation found for opportunity {opportunity_id}")
        conversation = max(conversations, key=lambda c: c.occurred_at)  # step 1

        claims = await self._claim_repo.list_claims_for_conversation(workspace_id, conversation.conversation_id)
        objection_claim = _select_objection_claim(claims)  # step 2
        if objection_claim is None:
            raise NoObjectionFoundError(
                f"no affirmed, non-rejected buyer objection found in conversation {conversation.conversation_id}"
            )

        evidence_text = ""
        if objection_claim.source_segment_id:
            segment = await self._conversation_repo.get_segment(workspace_id, objection_claim.source_segment_id)
            if segment:
                evidence_text = segment.text[
                    objection_claim.evidence_char_start:objection_claim.evidence_char_end
                ]

        objection_category = (objection_claim.object_value or "").strip().lower()
        all_assets = await self._content_repo.list_content_assets(workspace_id)
        matching = [
            (asset, [t for t in asset.tags if t.lower() == objection_category])
            for asset in all_assets
        ]
        matching = [(asset, tags) for asset, tags in matching if tags]  # step 3

        viewed_ids = await self._content_repo.list_viewed_asset_ids(workspace_id, buyer_contact_id)
        excluded = [a.content_asset_id for a, _ in matching if a.content_asset_id in viewed_ids]
        remaining = [(a, tags) for a, tags in matching if a.content_asset_id not in viewed_ids]  # step 4

        ranked = sorted(
            (RankedAsset(asset=a, matched_tags=tags, rank_score=float(len(tags))) for a, tags in remaining),
            key=lambda r: (-r.rank_score, r.asset.title),
        )  # step 5

        recommended = ranked[0].asset if ranked else None  # step 6

        explanation = (
            f"Objection '{objection_category}' raised (Claim {objection_claim.claim_id}) by a buyer "
            f"stakeholder in conversation {conversation.conversation_id} (most recent call for this "
            f"opportunity). {len(matching)} ContentAsset(s) tagged '{objection_category}' found via "
            f"curated tags; {len(excluded)} excluded as already viewed by the buyer; "
            f"{'top ranked by matched-tag count: ' + recommended.title if recommended else 'no unviewed asset available'}."
        )

        return ObjectionContentRecommendation(
            opportunity_id=opportunity_id,
            conversation_id=conversation.conversation_id,
            objection_claim=objection_claim,
            evidence_text=evidence_text,
            recommended_asset=recommended,
            ranked_candidates=ranked,
            excluded_viewed_asset_ids=excluded,
            mapping_source=MAPPING_SOURCE,
            explanation=explanation,
        )
