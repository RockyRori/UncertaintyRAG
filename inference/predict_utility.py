import pickle
from typing import List
import torch

from config import TFIDF_VECTORIZER_PATH, UTILITY_MODEL_PATH, HIDDEN_DIM, DROPOUT
from models.utility_predictor import UtilityMLP


class UtilityPredictor:
    def __init__(self):
        with open(TFIDF_VECTORIZER_PATH, "rb") as f:
            self.vectorizer = pickle.load(f)

        input_dim = len(self.vectorizer.get_feature_names_out())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = UtilityMLP(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(UTILITY_MODEL_PATH, map_location=self.device)
        )
        self.model.eval()

    @staticmethod
    def build_text(question: str, passage: str) -> str:
        return f"{question} [SEP] {passage}"

    def predict_one(self, question: str, passage: str) -> float:
        text = self.build_text(question, passage)
        x = self.vectorizer.transform([text]).toarray()
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(x_tensor)
            prob = torch.sigmoid(logits).item()

        return float(prob)

    def predict_batch(self, question: str, passages: List[str]) -> List[float]:
        return [self.predict_one(question, p) for p in passages]