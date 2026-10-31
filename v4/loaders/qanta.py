"""QANTA / Quizbowl (Rodriguez et al. 2019, "Quizbowl: The Case for Incremental Question Answering"), from the Hugging
Face release `qanta`, config "mode=full,char_skip=25" (one row per question, `full_question` = the complete pyramidal
question text), parquet files read directly.

Folds: guesstrain, buzztrain -> train; guessdev, buzzdev -> validation; guesstest, buzztest, adversarial -> test.
Subsets = every non-empty `category` (Current Events, Fine Arts, Geography, History, Literature, Mythology,
Philosophy, Religion, Science, Social Science, Trash). Rows with an empty category (only the 1145 `adversarial`
questions, which carry no category label) belong to no subset and are dropped (counted in the manifest).
Task short: question = full_question, answer = `page` with underscores as spaces (the Wikipedia title that is the
release's canonical answer), aliases = that title, the raw `page`/`answer` id, and `raw_answer` (the original answer
line with its accept/prompt directives) - de-duplicated.
Native id candidates = `qanta_id` (the release's own id) and the row index within its fold. The question hash is
checked on full_question and on `first_sentence` (the same question's opening clue).
"""
import pandas as pd

from rowkit import check, finish, hf_file

BENCH, FAMILY = "qanta", "qanta"
REPO, REV = "qanta", "e3c5602229e5c0c8572636a81cb366eb9ccfa890"
SOURCE = f"{REPO}@{REV} (config mode=full,char_skip=25)"
FOLDS = {"guesstrain": "train", "buzztrain": "train", "guessdev": "validation", "buzzdev": "validation",
         "guesstest": "test", "buzztest": "test", "adversarial": "test"}


def main():
    cands, no_cat = {}, 0
    for fold, split in FOLDS.items():
        df = pd.read_parquet(hf_file(REPO, f"mode=full,char_skip=25/{fold}-00000-of-00001.parquet", REV, BENCH))
        for i, r in enumerate(df.itertuples(index=False)):
            cat = (r.category or "").strip()
            if not cat:
                no_cat += 1
                continue
            title = r.page.replace("_", " ").strip()
            aliases = list(dict.fromkeys(a for a in (title, r.page, r.answer, (r.raw_answer or "").strip()) if a))
            row = {
                "id": f"{BENCH}/{cat}/{r.qanta_id}", "benchmark": BENCH, "subset": cat, "split": split,
                "task": "short", "context": "", "question": r.full_question, "options": [], "answer": title,
                "aliases": aliases,
                "meta": {"qanta_id": int(r.qanta_id), "fold": fold, "page": r.page, "raw_answer": r.raw_answer,
                         "subcategory": r.subcategory, "tournament": r.tournament, "difficulty": r.difficulty,
                         "year": int(r.year), "dataset": r.dataset, "qdb_id": int(r.qdb_id),
                         "proto_id": r.proto_id, "row_index_in_fold": i},
                "source": SOURCE,
            }
            rule = check([r.full_question, r.first_sentence], FAMILY, [int(r.qanta_id), i])
            cands.setdefault(cat, []).append((row, rule))
    finish(BENCH, FAMILY, SOURCE, cands, extra={
        "dropped_no_category": no_cat,
        "fold_to_split": FOLDS,
        "native_id": "qanta_id, and row index within fold",
        "question_hash_texts": "full_question and first_sentence",
    })


if __name__ == "__main__":
    main()
