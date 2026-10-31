"""WMT19 news translation, from the Hugging Face release wmt/wmt19 (parquet files read directly).

Source facts (dataset card / parquet): 9 configs cs-en, de-en, fi-en, fr-de, gu-en, kk-en, lt-en, ru-en, zh-en, each with
`train` (large parallel training corpora, not a test/dev set) and `validation` (the official WMT news dev/test set
shipped as the release's evaluation split). Each row is {"translation": {<lang>: str, <lang>: str}}.
We use only `validation` (the official evaluation sentences; the train split is web/parallel training data).

Subsets = every language pair in BOTH directions the release ships = 18 subsets named "<src>-<tgt>"
(e.g. "cs-en" = Czech->English, "en-cs" = English->Czech). The same row index yields one candidate per direction.
Task translate: question = source sentence, answer = reference translation, meta.src_lang / meta.tgt_lang.
Native id = row index within the config's validation split (the release has no ids); exclusion checks
sha256("wmt19:<row index>") and sha256(norm(source sentence)).
"""
import pandas as pd

from rowkit import check, finish, hf_file

BENCH, FAMILY = "wmt19", "wmt19"
REPO, REV = "wmt/wmt19", "c8a87a86736e7bf10c74c22e9033eba9146774e9"
SOURCE = f"{REPO}@{REV}"
CONFIGS = ["cs-en", "de-en", "fi-en", "fr-de", "gu-en", "kk-en", "lt-en", "ru-en", "zh-en"]
NAMES = {"cs": "Czech", "de": "German", "en": "English", "fi": "Finnish", "fr": "French", "gu": "Gujarati",
         "kk": "Kazakh", "lt": "Lithuanian", "ru": "Russian", "zh": "Chinese"}


def main():
    cands = {}
    for cfg in CONFIGS:
        a, b = cfg.split("-")
        df = pd.read_parquet(hf_file(REPO, f"{cfg}/validation-00000-of-00001.parquet", REV, BENCH))
        for i, tr in enumerate(df["translation"]):
            for s, t in ((a, b), (b, a)):
                sub = f"{s}-{t}"
                src, ref = str(tr[s]), str(tr[t])
                row = {
                    "id": f"{BENCH}/{sub}/{i}", "benchmark": BENCH, "subset": sub, "split": "validation",
                    "task": "translate", "context": "", "question": src, "options": [], "answer": ref,
                    "aliases": [],
                    "meta": {"src_lang": s, "tgt_lang": t, "src_lang_name": NAMES[s], "tgt_lang_name": NAMES[t],
                             "config": cfg, "row_index": i},
                    "source": SOURCE,
                }
                cands.setdefault(sub, []).append((row, check([src], FAMILY, [i])))
    finish(BENCH, FAMILY, SOURCE, cands, extra={
        "split_used": "validation (official WMT news dev/test sentences shipped by the release; train = parallel corpora, unused)",
        "native_id": "row index within <config>/validation",
    })


if __name__ == "__main__":
    main()
