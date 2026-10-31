"""ArcMMLU (Zhang et al. 2023, arXiv 2311.18658) from the authors' HF release patrickshitou/ArcMMLU
(linked from github.com/stzhang-patrick/ArcMMLU; CMMLU-format CSVs, Chinese).
Subsets = archive, data_science, information, library. Splits: dev (preferred) then test.
Native id = the CSV's own unnamed index column. Task mc: options A-D, answer = index of the `Answer` letter."""
import pandas as pd

from mc_runner import finish, hf_file, row

BENCH, FAMILY = "arcmmlu", "arcmmlu"
REPO, REV = "patrickshitou/ArcMMLU", "5c2eccb5cfb586a5ac9176c3685ab581545bd230"
SOURCE = f"{REPO}@{REV}"
SUBSETS = ["archive", "data_science", "information", "library"]


def main():
    cands = {}
    for sub in SUBSETS:
        for split in ["dev", "test"]:
            df = pd.read_csv(hf_file(REPO, f"{split}/{sub}.csv", REV, BENCH), dtype=str, keep_default_na=False)
            idcol = df.columns[0]
            assert idcol.startswith("Unnamed") or idcol == "", idcol
            for r in df.to_dict("records"):
                nid = int(r[idcol])
                opts = [r[k] for k in "ABCD"]
                ans = r["Answer"].strip()
                assert ans in "ABCD" and len(ans) == 1, (sub, split, nid, ans)
                out = row(BENCH, sub, f"{split}:{nid}", split, "mc", r["Question"], opts, "ABCD".index(ans), SOURCE,
                          meta={"row_index": nid, "language": "zh"})
                cands.setdefault(sub, []).append({"row": out, "native_id": nid, "qtexts": [r["Question"]]})
    finish(BENCH, FAMILY, SOURCE, cands)


if __name__ == "__main__":
    main()
