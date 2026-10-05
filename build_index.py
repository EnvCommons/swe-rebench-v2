"""Build task_index.json from the SWE-rebench-V2 parquet files.

Reads only the install_config column to determine which rows have a valid
test_cmd, then writes a lightweight index file that the environment
server loads at startup — so it never needs to scan the full dataset.

Usage:
    python build_index.py [--data-dir /path/to/parquet] [--output task_index.json]
"""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def build_index(data_dir: Path) -> dict:
    """Scan parquet files and return the index structure."""
    paths = sorted(data_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No .parquet files found in {data_dir}")

    # Build file layout from metadata only (no data read)
    files = []
    cum = 0
    for p in paths:
        n = pq.ParquetFile(p).metadata.num_rows
        files.append({"path": p.name, "offset": cum, "num_rows": n})
        cum += n
    total_rows = cum

    # Read only install_config column to filter valid rows
    # A task is valid if it has a test_cmd (required to evaluate patches).
    table = pq.read_table(paths, columns=["install_config"])
    col = table.column("install_config")
    valid_indices = []
    for i in range(len(col)):
        ic = col[i].as_py()
        if isinstance(ic, str):
            ic = json.loads(ic)
        if isinstance(ic, dict) and ic.get("test_cmd"):
            valid_indices.append(i)
    del table, col

    return {
        "total_rows": total_rows,
        "num_valid": len(valid_indices),
        "files": files,
        "valid_indices": valid_indices,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("/orwd_data"),
        help="Directory containing the parquet files",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("task_index.json"),
        help="Where to write the index",
    )
    args = parser.parse_args()

    print(f"Scanning {args.data_dir} ...")
    index = build_index(args.data_dir)
    args.output.write_text(json.dumps(index))
    print(
        f"Done. {index['num_valid']}/{index['total_rows']} valid tasks "
        f"written to {args.output}"
    )


if __name__ == "__main__":
    main()