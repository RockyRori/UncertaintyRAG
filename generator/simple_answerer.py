from generator.qa_generator import QAGenerator


class SimpleAnswerer:
    def __init__(
        self,
        model_name: str,
        max_input_length: int = 512,
        max_new_tokens: int = 32,
    ):
        self.available = True
        self.reason = ""
        self.generator = None

        try:
            self.generator = QAGenerator(
                model_name=model_name,
                max_input_length=max_input_length,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            self.available = False
            self.reason = str(e)

    def answer(self, question: str, passages: list[str]) -> str:
        if self.available and self.generator is not None:
            return self.generator.answer_with_passages(question, passages)

        # fallback：非常弱，只是防止系统直接崩
        if passages:
            return passages[0].split(".")[0].strip()
        return "I don't know."

    def answer_per_passage(self, question: str, passages: list[str]) -> list[str]:
        if self.available and self.generator is not None:
            return [
                self.generator.answer_with_single_passage(question, p)
                for p in passages
            ]

        return [p.split(".")[0].strip() for p in passages]