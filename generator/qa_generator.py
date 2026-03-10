from transformers import pipeline


class QAGenerator:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.pipe = pipeline(
            "text2text-generation",
            model=model_name,
            device=0
        )

    def build_prompt(self, question: str, contexts: list[str]) -> str:
        context_text = "\n".join([f"Context {i+1}: {c}" for i, c in enumerate(contexts)])
        prompt = (
            f"Answer the question based on the given contexts.\n\n"
            f"{context_text}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        return prompt

    def generate(self, question: str, contexts: list[str], max_new_tokens: int = 32):
        prompt = self.build_prompt(question, contexts)
        result = self.pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        return result[0]["generated_text"]