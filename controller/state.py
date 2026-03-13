from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class DecisionState:
    question: str
    gold_answers: List[str]

    evidence: List[Dict[str, Any]] = field(default_factory=list)
    utilities: List[float] = field(default_factory=list)
    candidate_answers: List[str] = field(default_factory=list)

    retrieval_uncertainty: float = 1.0
    conflict_uncertainty: float = 1.0
    stability_uncertainty: float = 1.0
    total_uncertainty: float = 1.0

    prev_total_uncertainty: Optional[float] = None
    delta_uncertainty: float = 0.0

    prev_best_utility: float = 0.0
    best_utility: float = 0.0
    evidence_gain: float = 0.0

    remaining_budget: int = 0
    step: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)

    last_action: Optional[str] = None
    stop_reason: str = ""

    final_action: Optional[str] = None
    final_answer: str = ""
    correct: int = 0