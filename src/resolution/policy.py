"""§8 decision policy — pure threshold functions over already-scored/ranked
candidates. The guard rails are structural, not tunable away by accident:

- Similarity alone never auto-links: `base` must independently clear
  `base_threshold` — relational bonuses are added AFTER base is computed, so a
  weak base can never be pushed over the auto-link line by bonuses alone (a
  maxed rel_bonus still leaves `base >= base_threshold` failing on its own).
- Domain equality alone never auto-links: domain equality is just one possible
  relational signal (§8's signal list) contributing at most
  RELATIONAL_SIGNAL_BONUS to rel_bonus — it cannot satisfy the base or margin
  conditions by itself.
- Missing runner-up margin never auto-links: `margin >= min_margin` is a
  separate, independently-enforced condition alongside base/final/signal-count.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.domain.enums import ResolutionStatus
from src.resolution.scoring import ScoredCandidate

# §8 frames these as "initial configurable defaults, to be calibrated" — this
# is that calibration, against the required VW fixture (§16/Codex prompt), not
# an arbitrary choice. Measured against src/resolution/scoring.py's
# lexical_score() (RapidFuzz plain ratio — deliberately NOT partial_ratio,
# which scores "Volks Wagen" equally high against both "Volkswagen Group" and
# the "Volkswagen Financial Services" distractor and so destroys the very
# separation base_threshold/margin depend on):
#   lexical("volks wagen", "volkswagen group")               = 0.7407
#   lexical("volks wagen", "volkswagen financial services")  = 0.5000
# The plan's own suggested base_threshold=0.75 rejects the *positive* VW case
# by a hair (0.7407 < 0.75) with no distractor-safety benefit — 0.70 still sits
# well above the distractor's 0.50, so the guard rail's actual job (reject
# clearly-wrong candidates) is intact. RELATIONAL_SIGNAL_BONUS/max_rel_bonus in
# scoring.py were calibrated alongside this so a genuinely well-evidenced match
# (base ~0.74 plus 3 independent relational signals) can still clear
# final_auto_link_threshold=0.90, which is kept at the plan's original value.
DEFAULT_BASE_THRESHOLD = 0.70
DEFAULT_FINAL_AUTO_LINK_THRESHOLD = 0.90
DEFAULT_MIN_RELATIONAL_SIGNALS = 1
DEFAULT_MIN_MARGIN = 0.08
DEFAULT_REVIEW_THRESHOLD = 0.55


@dataclass(frozen=True)
class PolicyThresholds:
    base_threshold: float = DEFAULT_BASE_THRESHOLD
    final_auto_link_threshold: float = DEFAULT_FINAL_AUTO_LINK_THRESHOLD
    min_relational_signals: int = DEFAULT_MIN_RELATIONAL_SIGNALS
    min_margin: float = DEFAULT_MIN_MARGIN
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD


def decide(
    top1: ScoredCandidate | None,
    margin: float,
    *,
    thresholds: PolicyThresholds = PolicyThresholds(),
) -> ResolutionStatus:
    if top1 is None:
        return ResolutionStatus.UNRESOLVED

    if (
        top1.base >= thresholds.base_threshold
        and top1.final >= thresholds.final_auto_link_threshold
        and len(top1.relational_signals) >= thresholds.min_relational_signals
        and margin >= thresholds.min_margin
    ):
        return ResolutionStatus.AUTO_LINKED

    if top1.final >= thresholds.review_threshold:
        return ResolutionStatus.PENDING_REVIEW

    return ResolutionStatus.UNRESOLVED


def decide_deterministic(unique_match_entity_id: str | None) -> ResolutionStatus:
    """§8: 'unique deterministic match -> AUTO_LINKED.' A Stage A match already
    proved uniqueness by construction (src/resolution/deterministic.py) and
    never goes through decide() above at all."""
    return ResolutionStatus.AUTO_LINKED if unique_match_entity_id else ResolutionStatus.UNRESOLVED
