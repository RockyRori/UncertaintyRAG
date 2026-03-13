from dataclasses import dataclass, field
from typing import List, Dict, Any


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

    remaining_budget: int = 0
    step: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)

    final_action: str | None = None
    final_answer: str = ""
    correct: int = 0