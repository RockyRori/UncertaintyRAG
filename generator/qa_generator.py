import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class QAGenerator:
    def __init__(self, model_name: str, device: str = "cuda", max_input_length: int = 512, max_new_tokens: int = 32):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def _build_full_prompt(self, question: str, passages: list[str]) -> str:
        knowledge = "\n".join([f"[{i+1}] {p}" for i, p in enumerate(passages)])
        prompt = (
            "Answer the question based on the knowledge below.\n"
            "Give a very short answer phrase.\n\n"
            f"Knowledge:\n{knowledge}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return prompt

    def _build_single_prompt(self, question: str, passage: str) -> str:
        prompt = (
            "Answer the question based only on the passage below.\n"
            "Give a very short answer phrase.\n\n"
            f"Passage: {passage}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return prompt

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text.strip()

    def answer_with_passages(self, question: str, passages: list[str]) -> str:
        prompt = self._build_full_prompt(question, passages)
        return self._generate(prompt)

    def answer_with_single_passage(self, question: str, passage: str) -> str:
        prompt = self._build_single_prompt(question, passage)
        return self._generate(prompt)