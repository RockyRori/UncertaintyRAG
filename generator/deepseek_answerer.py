import argparse
import os
import sys
from pathlib import Path

from openai import OpenAI

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.text_utils import postprocess_answer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEEPSEEK_API_KEY_PLACEHOLDER = "YOUR_DEEPSEEK_API_KEY_MUST_IN_ENV"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def load_env_file(env_path: Path = PROJECT_ROOT / ".env") -> None:
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_env_file()


def is_placeholder_api_key(api_key: str) -> bool:
    return api_key.strip().upper().startswith("YOUR_DEEPSEEK_API_KEY")


class DeepSeekAnswerer:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-v4-flash",
        system_prompt: str = (
            "You are a careful short-answer QA extractor. "
            "Return only the shortest answer span supported by the provided evidence. "
            "Do not explain, cite sources, add prefixes, or write a full sentence unless "
            "the answer itself must be a full sentence. If the evidence is insufficient, "
            "return exactly: I don't know."
        ),
        reasoning_effort: str = "medium",
        thinking_enabled: bool = False,
    ):
        self.api_key = (
            api_key
            or os.environ.get("DEEPSEEK_API_KEY")
            or DEEPSEEK_API_KEY_PLACEHOLDER
        )
        self.model = model
        self.system_prompt = system_prompt
        self.reasoning_effort = reasoning_effort
        self.thinking_enabled = thinking_enabled
        self.client = OpenAI(api_key=self.api_key, base_url=DEEPSEEK_BASE_URL)

    def _ensure_api_key(self) -> None:
        if is_placeholder_api_key(self.api_key):
            raise RuntimeError(
                "Please set DEEPSEEK_API_KEY or replace "
                "DEEPSEEK_API_KEY_PLACEHOLDER with your DeepSeek API key."
            )

    def _build_prompt(self, question: str, passages: list[str]) -> str:
        if passages:
            knowledge = "\n".join(
                f"[{idx + 1}] {passage}" for idx, passage in enumerate(passages)
            )
            return (
                "Answer the question based on the knowledge below.\n"
                "Return only the shortest answer span. Do not explain.\n"
                "If the knowledge is insufficient, return exactly: I don't know.\n\n"
                f"Knowledge:\n{knowledge}\n\n"
                f"Question: {question}"
            )

        return question

    def _chat(self, user_prompt: str) -> str:
        self._ensure_api_key()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
            reasoning_effort=self.reasoning_effort,
            extra_body={
                "thinking": {
                    "type": "enabled" if self.thinking_enabled else "disabled"
                }
            },
        )
        return response.choices[0].message.content.strip()

    def answer(self, question: str, passages: list[str] | None = None) -> str:
        passages = passages or []
        raw_answer = self._chat(self._build_prompt(question, passages))
        return postprocess_answer(raw_answer, question)

    def answer_per_passage(self, question: str, passages: list[str]) -> list[str]:
        return [self.answer(question, [passage]) for passage in passages]

    def answer_with_passages(self, question: str, passages: list[str]) -> str:
        return self.answer(question, passages)

    def answer_with_single_passage(self, question: str, passage: str) -> str:
        return self.answer(question, [passage])


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask DeepSeek a question.")
    parser.add_argument("--question", default="Hello")
    parser.add_argument(
        "--passage",
        action="append",
        default=[],
        help="Optional evidence passage. Repeat this argument to pass more evidence.",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="deepseek-v4-flash")
    args = parser.parse_args()

    answerer = DeepSeekAnswerer(api_key=args.api_key, model=args.model)
    print(answerer.answer(args.question, args.passage))


if __name__ == "__main__":
    main()
