from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class DecisionState:
    question: str
    gold_answers: List[str]

    evidence: List[Dict[str, Any]] = field(default_factory=list)
    utilities: List[float] = field(default_factory=list)
    candidate_answers: List[str] = field(default_factory=list)

    # ---- unified uncertainty components ----
    generation_entropy: float = 1.0          # H_t
    utility_uncertainty: float = 1.0         # 1 - u_t
    stability_score: float = 0.0             # S_t
    total_uncertainty: float = 1.0           # U_t = alpha H + beta(1-u) + gamma(1-S)

    # ---- compatibility with old metrics/logs ----
    retrieval_uncertainty: float = 1.0
    conflict_uncertainty: float = 1.0
    stability_uncertainty: float = 1.0

    # ---- utility-aware controller signals ----
    prev_total_uncertainty: Optional[float] = None
    delta_uncertainty: float = 0.0

    prev_best_utility: float = 0.0
    best_utility: float = 0.0
    evidence_gain: float = 0.0

    best_answer: str = ""
    prev_best_answer: str = ""
    best_answer_weight: float = 0.0

    answer_utility: float = 0.0
    continue_utility: float = 0.0
    service_utility: float = 0.0

    remaining_budget: int = 0
    step: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)

    last_action: Optional[str] = None
    stop_reason: str = ""

    final_action: Optional[str] = None
    final_answer: str = ""
    correct: int = 0