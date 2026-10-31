"""GeographyData: no official public release could be located (searched Hugging Face Hub dataset names
"geographydata"/"geography_data"/"geography-qa"/..., GitHub repositories and code, and the web for a paper or
dataset card of that name). The exclusion file implies a small set (ids 0..49). Rather than invent or substitute
data, this loader writes no data file and records the status in the manifest."""
from common import update_manifest

BENCH = "geographydata"


def main():
    info = {"benchmark": BENCH, "exclusion_family": "geographydata", "status": "source_not_found", "source": None,
            "subsets": [], "budget": 200, "available": 0, "excluded_question_hash": 0, "excluded_id_hash": 0,
            "usable": 0, "taken": 0, "shortfall": 200,
            "note": "no authoritative public release found (HF hub, GitHub, web); not substituted with another dataset"}
    update_manifest(BENCH, info)
    print(info)


if __name__ == "__main__":
    main()
