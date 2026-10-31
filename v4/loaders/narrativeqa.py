"""NarrativeQA (Kocisky et al., TACL 2018), from the authors' original release (github.com/deepmind/narrativeqa,
commit 904246f6d1fe99a99a08a03501fe3e619af2cee5), as mirrored byte-for-byte in the Hugging Face repo deepmind/narrativeqa
(data/narrativeqa-master.zip, a GitHub archive whose zip comment is that commit). CSVs are read with pandas only.

Files: qaps.csv (document_id, set, question, answer1, answer2, ...), third_party/wikipedia/summaries.csv
(document_id, set, summary, ...), documents.csv (document_id, set, kind, story_word_count, wiki_title, ...).
Context decision: the paper's "summaries only" setting. Full stories have median ~34k words (mean ~62k, max 430k),
far too long to send as a calibration prompt, so context = the Wikipedia summary in full.
Subsets = documents.csv `kind` (gutenberg = books, movie = movie scripts). Splits: train > valid > test (source `set`).
Task reading: answer = answer1, aliases = [answer1, answer2] (all references, de-duplicated).
Native id = row index of the question within its split in qaps.csv order (the release has no question ids); the
emitted id is "narrativeqa/<kind>/<split>-<index>". Exclusion checks sha256("narrativeqa:<index>") and the question hash.
"""
import io
import zipfile

import pandas as pd

from rowkit import check, finish, hf_file

BENCH, FAMILY = "narrativeqa", "narrativeqa"
REPO, REV = "deepmind/narrativeqa", "2e643e7363944af1c33a652d1c87320d0871c4e4"
GH_COMMIT = "904246f6d1fe99a99a08a03501fe3e619af2cee5"
SOURCE = f"{REPO}@{REV}:data/narrativeqa-master.zip (github.com/deepmind/narrativeqa@{GH_COMMIT})"
SPLIT = {"train": "train", "valid": "validation", "test": "test"}


def main():
    z = zipfile.ZipFile(hf_file(REPO, "data/narrativeqa-master.zip", REV, BENCH))
    assert z.comment.decode().strip() == GH_COMMIT, z.comment

    def csv(name):
        return pd.read_csv(io.BytesIO(z.read("narrativeqa-master/" + name)), keep_default_na=False, dtype=str)

    docs = csv("documents.csv").set_index("document_id")
    summ = csv("third_party/wikipedia/summaries.csv").set_index("document_id")
    qa = csv("qaps.csv")
    qa["idx"] = qa.groupby("set").cumcount()
    cands = {}
    for r in qa.itertuples(index=False):
        d = docs.loc[r.document_id]
        kind = d["kind"]
        refs = list(dict.fromkeys(a.strip() for a in (r.answer1, r.answer2) if a.strip()))
        row = {
            "id": f"{BENCH}/{kind}/{SPLIT[r.set]}-{r.idx}", "benchmark": BENCH, "subset": kind, "split": SPLIT[r.set],
            "task": "reading", "context": summ.loc[r.document_id, "summary"].strip(), "question": r.question,
            "options": [], "answer": refs[0], "aliases": refs,
            "meta": {"document_id": r.document_id, "kind": kind, "title": d["wiki_title"], "wiki_url": d["wiki_url"],
                     "story_url": d["story_url"], "story_word_count": int(d["story_word_count"]),
                     "context_type": "wikipedia_summary", "row_index_in_split": int(r.idx)},
            "source": SOURCE,
        }
        cands.setdefault(kind, []).append((row, check([r.question], FAMILY, [int(r.idx)])))
    finish(BENCH, FAMILY, SOURCE, cands, extra={
        "context_decision": "Wikipedia summary (summaries-only setting); full stories median ~34k words, too long",
        "native_id": "row index within split (qaps.csv order)",
    })


if __name__ == "__main__":
    main()
