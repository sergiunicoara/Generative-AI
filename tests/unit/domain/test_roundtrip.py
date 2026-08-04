"""Every model in src/domain/* must round-trip through model_dump_json() ->
model_validate_json() unchanged — proves P0's contracts are well-formed Pydantic
models with no lossy custom serialization, entirely without a database (this
phase's own exit criterion, docs/plan.md §16 P0 row).

st.builds(ModelClass) auto-infers a Hypothesis strategy per field from the
model's type annotations (str/int/float/datetime/enum/Optional all resolve
correctly — verified against this Hypothesis/Pydantic version pairing before
writing this file). Models with cross-field model_validators get explicit
overrides here so Hypothesis generates only combinations those validators
accept; that validator logic itself is covered by its own dedicated test
(test_claim_identity_split.py, test_mention_span_validation.py, etc.) — this
file is about round-trip fidelity, not validator correctness.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.domain.assertion import (
    Claim,
    Conflict,
    ErasureEvent,
    ExtractionRun,
    ResolutionDecision,
    ReviewDecision,
)
from src.domain.conversation import (
    Conversation,
    ExtractionWindow,
    Mention,
    Participant,
    SpeakerResolution,
    TranscriptSegment,
)
from src.domain.crm import (
    Account,
    Activity,
    Contact,
    Lead,
    Meeting,
    Opportunity,
    OpportunityContactRole,
    Seller,
    SourceRecord,
    SourceSnapshot,
)
from src.domain.knowledge import (
    ActionItem,
    AssetView,
    BuyingSignal,
    Blocker,
    Commitment,
    ContentAsset,
    Feature,
    Objection,
    PainPoint,
    Product,
    Share,
)
from src.domain.stakeholder import StakeholderAssignment

_safe_float = st.floats(allow_nan=False, allow_infinity=False)

MODEL_STRATEGIES: list[tuple[type, object]] = [
    # ── crm.py — no cross-field validators, plain inference ──────────────────
    (SourceRecord, st.builds(SourceRecord)),
    (SourceSnapshot, st.builds(SourceSnapshot)),
    (Account, st.builds(Account)),
    (Contact, st.builds(Contact)),
    (
        Lead,
        st.builds(Lead, converted_to_type=st.none(), converted_to_id=st.none()),
    ),
    (Seller, st.builds(Seller)),
    (Opportunity, st.builds(Opportunity)),
    (OpportunityContactRole, st.builds(OpportunityContactRole)),
    (Meeting, st.builds(Meeting)),
    (Activity, st.builds(Activity)),
    # ── conversation.py ────────────────────────────────────────────────────
    (Conversation, st.builds(Conversation)),
    (Participant, st.builds(Participant)),
    (TranscriptSegment, st.builds(TranscriptSegment)),
    (
        ExtractionWindow,
        st.builds(
            ExtractionWindow,
            segment_ids=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5),
            start_segment_index=st.just(0),
            end_segment_index=st.integers(min_value=0, max_value=50),
        ),
    ),
    (
        Mention,
        st.builds(
            Mention,
            char_start=st.just(0),
            char_end=st.integers(min_value=1, max_value=200),
        ),
    ),
    (SpeakerResolution, st.builds(SpeakerResolution)),
    # ── knowledge.py — no cross-field validators ───────────────────────────
    (Product, st.builds(Product)),
    (Feature, st.builds(Feature)),
    (Objection, st.builds(Objection)),
    (PainPoint, st.builds(PainPoint)),
    (Blocker, st.builds(Blocker)),
    (BuyingSignal, st.builds(BuyingSignal)),
    (ActionItem, st.builds(ActionItem)),
    (Commitment, st.builds(Commitment)),
    (ContentAsset, st.builds(ContentAsset)),
    (Share, st.builds(Share)),
    (AssetView, st.builds(AssetView)),
    # ── assertion.py ────────────────────────────────────────────────────────
    (
        Claim,
        st.builds(
            Claim,
            object_id=st.text(min_size=1, max_size=20),
            object_value=st.none(),
            evidence_char_start=st.just(0),
            evidence_char_end=st.integers(min_value=1, max_value=200),
            confidence=_safe_float.filter(lambda f: 0.0 <= f <= 1.0),
        ),
    ),
    (ExtractionRun, st.builds(ExtractionRun)),
    (
        ResolutionDecision,
        st.builds(
            ResolutionDecision,
            resolved_entity_id=st.text(min_size=1, max_size=20),
            lexical_score=st.none() | _safe_float,
            semantic_score=st.none() | _safe_float,
            base_score=st.none() | _safe_float,
            relational_bonus=st.none() | _safe_float,
            final_score=st.none() | _safe_float,
            margin=st.none() | _safe_float,
        ),
    ),
    (
        ReviewDecision,
        st.builds(
            ReviewDecision,
            rejected=st.just(False),
            selected_entity_id=st.text(min_size=1, max_size=20),
        ),
    ),
    (Conflict, st.builds(Conflict)),
    (ErasureEvent, st.builds(ErasureEvent)),
    # ── stakeholder.py ─────────────────────────────────────────────────────
    (StakeholderAssignment, st.builds(StakeholderAssignment)),
]


def test_all_domain_models_are_covered():
    """Guard against silently forgetting to add a new model to this suite."""
    import src.domain.assertion as assertion_mod
    import src.domain.conversation as conversation_mod
    import src.domain.crm as crm_mod
    import src.domain.knowledge as knowledge_mod
    import src.domain.stakeholder as stakeholder_mod
    from pydantic import BaseModel

    covered = {cls for cls, _ in MODEL_STRATEGIES}
    for module in (assertion_mod, conversation_mod, crm_mod, knowledge_mod, stakeholder_mod):
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseModel)
                and obj is not BaseModel
                and obj.__module__ == module.__name__
            ):
                assert obj in covered, f"{module.__name__}.{name} has no round-trip strategy"


def _make_roundtrip_check(model_cls, strategy):
    @given(strategy)
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def check(instance):
        dumped = instance.model_dump_json()
        restored = model_cls.model_validate_json(dumped)
        assert restored == instance

    return check


def test_all_models_roundtrip():
    for model_cls, strategy in MODEL_STRATEGIES:
        _make_roundtrip_check(model_cls, strategy)()
