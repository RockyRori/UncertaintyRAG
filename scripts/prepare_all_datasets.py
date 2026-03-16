import json
import re
from pathlib import Path
from typing import Any, Dict, List

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["squad", "nq", "triviaqa", "webq", "popqa"]


# =====================================
# 基础工具
# =====================================
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_json(data: Any, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_list(x) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def dedup_corpus(corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for item in corpus:
        title = normalize_text(item.get("title", ""))
        text = normalize_text(item.get("text", ""))
        if not text:
            continue
        key = (title, text)
        if key not in seen:
            seen.add(key)
            out.append({
                "id": str(item.get("id", f"doc_{len(out)}")),
                "title": title,
                "text": text
            })
    return out


def make_qa_item(qid: str, split: str, question: str, gold_answers: List[str]) -> Dict[str, Any]:
    return {
        "id": str(qid),
        "split": normalize_text(split).lower(),
        "question": normalize_text(question),
        "gold_answers": [normalize_text(a) for a in gold_answers if normalize_text(a)]
    }


def make_corpus_item(doc_id: str, text: str, title: str = "") -> Dict[str, Any]:
    return {
        "id": str(doc_id),
        "title": normalize_text(title),
        "text": normalize_text(text)
    }


def extract_text_from_nq_tokens(tokens: List[Dict[str, Any]]) -> str:
    """
    Natural Questions 的 document.tokens 常带 html token。
    这里只保留非 html 的 token，拼成正文。
    """
    words = []
    for t in tokens:
        token = t.get("token", "")
        is_html = t.get("is_html", False)
        if token and not is_html:
            words.append(token)
    return normalize_text(" ".join(words))


def summarize_stats(qa_items: List[Dict[str, Any]], corpus_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    split_count = {}
    for x in qa_items:
        split = x.get("split", "unknown")
        split_count[split] = split_count.get(split, 0) + 1

    return {
        "qa_count": len(qa_items),
        "corpus_count": len(corpus_items),
        "splits": split_count
    }


# =====================================
# 1. SQuAD
# raw:
#   squad_train.jsonl
#   squad_dev.jsonl
# =====================================
def prepare_squad():
    qa_items = []
    corpus_items = []
    doc_idx = 0

    sources = [
        ("train", RAW_DIR / "squad_train.jsonl"),
        ("test", RAW_DIR / "squad_dev.jsonl"),
    ]

    for split, path in sources:
        if not path.exists():
            print(f"[WARN] Missing {path}, skip SQuAD.")
            return None

        data = load_jsonl(path)
        for i, ex in enumerate(data):
            qid = ex.get("id", f"squad_{split}_{i}")
            question = ex.get("question", "")
            context = ex.get("context", "")
            title = ex.get("title", "")

            answers = ex.get("answers", {})
            gold_answers = []
            if isinstance(answers, dict):
                gold_answers = safe_list(answers.get("text", []))

            qa_items.append(make_qa_item(qid, split, question, gold_answers))
            corpus_items.append(
                make_corpus_item(
                    doc_id=f"squad_doc_{doc_idx}",
                    text=context,
                    title=title
                )
            )
            doc_idx += 1

    corpus_items = dedup_corpus(corpus_items)

    save_json(qa_items, PROCESSED_DIR / "squad_qa.json")
    save_json(corpus_items, PROCESSED_DIR / "squad_corpus.json")

    stats = summarize_stats(qa_items, corpus_items)
    print(f"[OK] SQuAD -> {stats}")
    return stats


# =====================================
# 2. Natural Questions
# raw:
#   nq.jsonl
# =====================================
def prepare_nq():
    path = RAW_DIR / "nq.jsonl"
    if not path.exists():
        print(f"[WARN] Missing {path}, skip NQ.")
        return None

    data = load_jsonl(path)
    qa_items = []
    corpus_items = []

    def flatten_to_strings(x):
        out = []
        if x is None:
            return out
        if isinstance(x, str):
            s = normalize_text(x)
            if s:
                out.append(s)
            return out
        if isinstance(x, (int, float, bool)):
            s = normalize_text(x)
            if s:
                out.append(s)
            return out
        if isinstance(x, list):
            for item in x:
                out.extend(flatten_to_strings(item))
            return out
        if isinstance(x, dict):
            # 常见情况：{"text": "..."} 或其他嵌套结构
            if "text" in x:
                out.extend(flatten_to_strings(x.get("text")))
            else:
                for v in x.values():
                    out.extend(flatten_to_strings(v))
            return out
        return out

    for i, ex in enumerate(data):
        qid = ex.get("id", f"nq_{i}")

        question_obj = ex.get("question", {})
        if isinstance(question_obj, dict):
            question = question_obj.get("text", "")
        else:
            question = question_obj

        gold_answers = []

        annotations = safe_list(ex.get("annotations", []))
        for ann in annotations:
            short_answers = safe_list(ann.get("short_answers", []))
            for sa in short_answers:
                gold_answers.extend(flatten_to_strings(sa))

        if not gold_answers:
            for ann in annotations:
                yes_no = ann.get("yes_no_answer", "NONE")
                if yes_no and yes_no != "NONE":
                    gold_answers.extend(flatten_to_strings(yes_no))

        # 去重
        gold_answers = list(dict.fromkeys([normalize_text(x) for x in gold_answers if normalize_text(x)]))

        document = ex.get("document", {})
        title = ""
        context = ""

        if isinstance(document, dict):
            title = document.get("title", "")
            tokens = safe_list(document.get("tokens", []))
            if tokens:
                context = extract_text_from_nq_tokens(tokens)

        if not context:
            answer_text = "; ".join(gold_answers[:5]) if gold_answers else ""
            context = f"Question: {question}. Candidate answers: {answer_text}"

        qa_items.append(make_qa_item(qid, "train", question, gold_answers))
        corpus_items.append(make_corpus_item(f"nq_doc_{i}", context, title))

    corpus_items = dedup_corpus(corpus_items)

    save_json(qa_items, PROCESSED_DIR / "nq_qa.json")
    save_json(corpus_items, PROCESSED_DIR / "nq_corpus.json")

    stats = summarize_stats(qa_items, corpus_items)
    print(f"[OK] NQ -> {stats}")
    return stats

# =====================================
# 3. TriviaQA
# raw:
#   trivia_train.jsonl
#   trivia_dev.jsonl
# =====================================
def prepare_triviaqa():
    qa_items = []
    corpus_items = []
    doc_idx = 0

    sources = [
        ("train", RAW_DIR / "trivia_train.jsonl"),
        ("test", RAW_DIR / "trivia_dev.jsonl"),
    ]

    for split, path in sources:
        if not path.exists():
            print(f"[WARN] Missing {path}, skip TriviaQA.")
            return None

        data = load_jsonl(path)
        for i, ex in enumerate(data):
            qid = ex.get("question_id", f"triviaqa_{split}_{i}")
            question = ex.get("question", "")

            answer_obj = ex.get("answer", {})
            gold_answers = []
            if isinstance(answer_obj, dict):
                value = answer_obj.get("value", "")
                aliases = safe_list(answer_obj.get("aliases", []))
                if value:
                    gold_answers.append(value)
                gold_answers.extend(aliases)

            # 去重
            gold_answers = list(dict.fromkeys([normalize_text(x) for x in gold_answers if normalize_text(x)]))

            # TriviaQA 的字段比较飘，优先找 search_results / entity_pages
            context_parts = []

            search_results = ex.get("search_results", {})
            if isinstance(search_results, dict):
                for k, v in search_results.items():
                    if isinstance(v, list):
                        if k.lower() in {"search_context", "description", "snippet", "search_results"}:
                            context_parts.extend([normalize_text(x) for x in v if normalize_text(x)])

            entity_pages = ex.get("entity_pages", {})
            if isinstance(entity_pages, dict):
                for k, v in entity_pages.items():
                    if isinstance(v, list):
                        if k.lower() in {"wiki_context", "description"}:
                            context_parts.extend([normalize_text(x) for x in v if normalize_text(x)])

            context = normalize_text(" ".join(context_parts))

            if not context:
                context = f"Question: {question}. Candidate answers: {'; '.join(gold_answers[:5])}"

            qa_items.append(make_qa_item(qid, split, question, gold_answers))
            corpus_items.append(
                make_corpus_item(
                    doc_id=f"triviaqa_doc_{doc_idx}",
                    text=context,
                    title="TriviaQA"
                )
            )
            doc_idx += 1

    corpus_items = dedup_corpus(corpus_items)

    save_json(qa_items, PROCESSED_DIR / "triviaqa_qa.json")
    save_json(corpus_items, PROCESSED_DIR / "triviaqa_corpus.json")

    stats = summarize_stats(qa_items, corpus_items)
    print(f"[OK] TriviaQA -> {stats}")
    return stats


# =====================================
# 4. WebQuestions
# raw:
#   webq_train.jsonl
#   webq_test.jsonl
# =====================================
def prepare_webq():
    qa_items = []
    corpus_items = []
    doc_idx = 0

    sources = [
        ("train", RAW_DIR / "webq_train.jsonl"),
        ("test", RAW_DIR / "webq_test.jsonl"),
    ]

    for split, path in sources:
        if not path.exists():
            print(f"[WARN] Missing {path}, skip WebQuestions.")
            return None

        data = load_jsonl(path)
        for i, ex in enumerate(data):
            qid = ex.get("url", f"webq_{split}_{i}")
            question = ex.get("question", "")

            answers = safe_list(ex.get("answers", []))
            gold_answers = [normalize_text(x) for x in answers if normalize_text(x)]

            # WebQuestions 通常没有标准 context，构造 pseudo corpus
            pseudo_context = f"Question: {question}. Reference answers: {'; '.join(gold_answers[:5])}"

            qa_items.append(make_qa_item(qid, split, question, gold_answers))
            corpus_items.append(
                make_corpus_item(
                    doc_id=f"webq_doc_{doc_idx}",
                    text=pseudo_context,
                    title="WebQuestions"
                )
            )
            doc_idx += 1

    corpus_items = dedup_corpus(corpus_items)

    save_json(qa_items, PROCESSED_DIR / "webq_qa.json")
    save_json(corpus_items, PROCESSED_DIR / "webq_corpus.json")

    stats = summarize_stats(qa_items, corpus_items)
    print(f"[OK] WebQ -> {stats}")
    return stats


# =====================================
# 5. PopQA
# raw:
#   popqa_test.jsonl
# =====================================
def prepare_popqa():
    path = RAW_DIR / "popqa_test.jsonl"
    if not path.exists():
        print(f"[WARN] Missing {path}, skip PopQA.")
        return None

    data = load_jsonl(path)
    qa_items = []
    corpus_items = []

    for i, ex in enumerate(data):
        qid = ex.get("id", f"popqa_{i}")
        question = ex.get("question", "")

        possible_answers = safe_list(ex.get("possible_answers", []))
        gold_answers = [normalize_text(x) for x in possible_answers if normalize_text(x)]

        subject = ex.get("subject", "")
        prop = ex.get("prop", "")
        obj = ex.get("object", "")

        pseudo_context = (
            f"Subject: {normalize_text(subject)}. "
            f"Relation: {normalize_text(prop)}. "
            f"Object: {normalize_text(obj)}. "
            f"Question: {normalize_text(question)}. "
            f"Candidate answers: {'; '.join(gold_answers[:5])}"
        )

        qa_items.append(make_qa_item(qid, "test", question, gold_answers))
        corpus_items.append(make_corpus_item(f"popqa_doc_{i}", pseudo_context, "PopQA"))

    corpus_items = dedup_corpus(corpus_items)

    save_json(qa_items, PROCESSED_DIR / "popqa_qa.json")
    save_json(corpus_items, PROCESSED_DIR / "popqa_corpus.json")

    stats = summarize_stats(qa_items, corpus_items)
    print(f"[OK] PopQA -> {stats}")
    return stats


# =====================================
# 汇总
# =====================================
def main():
    all_stats = {}

    squad_stats = prepare_squad()
    if squad_stats is not None:
        all_stats["squad"] = squad_stats

    nq_stats = prepare_nq()
    if nq_stats is not None:
        all_stats["nq"] = nq_stats

    trivia_stats = prepare_triviaqa()
    if trivia_stats is not None:
        all_stats["triviaqa"] = trivia_stats

    webq_stats = prepare_webq()
    if webq_stats is not None:
        all_stats["webq"] = webq_stats

    popqa_stats = prepare_popqa()
    if popqa_stats is not None:
        all_stats["popqa"] = popqa_stats

    save_json(all_stats, PROCESSED_DIR / "all_datasets_stats.json")

    print("\n[OK] Saved summary stats to data/processed/all_datasets_stats.json")
    print("[DONE] All available datasets have been processed.")


if __name__ == "__main__":
    main()