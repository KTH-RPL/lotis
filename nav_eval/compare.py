"""Compare two navigation evaluation result directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def episode_key(item):
    return (item["scene_id"], item["trajectory_id"], item["query_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two nav_eval results.json files or directories")
    parser.add_argument("--reference", required=True, help="Reference results.json or result directory")
    parser.add_argument("--candidate", required=True, help="Candidate results.json or result directory")
    parser.add_argument("--max-diffs", type=int, default=20)
    args = parser.parse_args()

    ref_path = _result_path(args.reference)
    cand_path = _result_path(args.candidate)
    ref = json.loads(ref_path.read_text())
    cand = json.loads(cand_path.read_text())

    print("Summary")
    for key in ("num_trajectories", "success_rate", "mean_spl", "success_spl", "timeout_rate", "mean_path_length", "mean_steps"):
        r = ref["summary"].get(key)
        c = cand["summary"].get(key)
        if isinstance(r, (int, float)) and isinstance(c, (int, float)):
            print(f"  {key}: ref={r:.6f} cand={c:.6f} delta={c-r:+.6f}")
        else:
            print(f"  {key}: ref={r} cand={c}")

    ref_eps = {episode_key(item): item for item in ref["trajectories"]}
    cand_eps = {episode_key(item): item for item in cand["trajectories"]}
    missing = sorted(set(ref_eps) - set(cand_eps))
    extra = sorted(set(cand_eps) - set(ref_eps))
    print(f"\nEpisode IDs: matched={len(set(ref_eps) & set(cand_eps))} missing={len(missing)} extra={len(extra)}")
    if missing:
        print("  missing:", missing[: args.max_diffs])
    if extra:
        print("  extra:", extra[: args.max_diffs])

    diffs = []
    for key in sorted(set(ref_eps) & set(cand_eps)):
        r = ref_eps[key]
        c = cand_eps[key]
        if r["success"] != c["success"]:
            diffs.append((key, "success", r["success"], c["success"]))
        elif abs(r["spl"] - c["spl"]) > 1e-6 or abs(r["path_length"] - c["path_length"]) > 1e-6:
            diffs.append((key, "metric", r["spl"], c["spl"]))

    print(f"Per-episode differences: {len(diffs)}")
    for diff in diffs[: args.max_diffs]:
        print(" ", diff)


def _result_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_dir():
        path = path / "results.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


if __name__ == "__main__":
    main()
