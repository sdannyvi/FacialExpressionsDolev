"""
convert_paths.py - make the file_path column in CSV files relative.

Turns absolute paths like
    /gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer2013/PrivateTest/angry/x.png
into
    data/fer2013/PrivateTest/angry/x.png

by cutting everything before the "data" folder component.

Usage
-----
  # 1. look first (dry run, nothing is written):
  python convert_paths.py ablation
  python convert_paths.py ablation --recursive

  # 2. when the preview looks right, add --apply:
  python convert_paths.py ablation --recursive --apply

Options
-------
  --recursive        also process CSVs in subfolders (off by default)
  --apply            actually write the files (off by default = dry run)
  --backup           keep the original as <name>.csv.bak when applying
  --column NAME      column to convert (default: file_path)
  --marker NAME      folder to start the relative path from (default: data)
"""

import argparse
import csv
import os
import sys

# CSV fields can be long; raise the limit so nothing is silently truncated.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def to_relative(value, marker):
    """
    Return (new_value, status).

    status is one of:
      "converted"       - path was cut at the marker folder
      "already"         - path already starts with the marker folder
      "no_marker"       - marker folder not found in the path -> left untouched
      "empty"           - blank value -> left untouched
      "marker_repeated" - marker appears more than once -> left untouched
    """
    if value is None or value.strip() == "":
        return value, "empty"

    # Normalise Windows backslashes so both path styles work.
    parts = value.replace("\\", "/").split("/")

    hits = [i for i, p in enumerate(parts) if p == marker]

    if not hits:
        return value, "no_marker"
    if len(hits) > 1:
        # Ambiguous - refuse to guess, report it instead.
        return value, "marker_repeated"

    idx = hits[0]
    new_value = "/".join(parts[idx:])

    if idx == 0:
        return new_value, "already"
    return new_value, "converted"


def process_file(path, column, marker, apply_changes, backup):
    """Convert one CSV. Returns True if it was (or would be) changed."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print(f"  SKIP  {os.path.basename(path)}  (empty file)")
            return False

        if column not in header:
            print(f"  SKIP  {os.path.basename(path)}  (no '{column}' column)")
            return False

        col_idx = header.index(column)
        rows = list(reader)

    counts = {}
    examples = []
    for row in rows:
        if col_idx >= len(row):          # short/ragged row - leave alone
            counts["short_row"] = counts.get("short_row", 0) + 1
            continue
        new_value, status = to_relative(row[col_idx], marker)
        counts[status] = counts.get(status, 0) + 1
        if status == "converted":
            if len(examples) < 1:
                examples.append((row[col_idx], new_value))
            row[col_idx] = new_value

    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"  {os.path.basename(path)}  ({len(rows)} rows: {summary})")
    for before, after in examples:
        print(f"      before: {before}")
        print(f"      after : {after}")

    problems = counts.get("no_marker", 0) + counts.get("marker_repeated", 0)
    if problems:
        print(f"      !! {problems} row(s) left untouched - check these manually")

    if counts.get("converted", 0) == 0:
        return False

    if not apply_changes:
        return True

    # Detect the original line ending so we write the file back the same way.
    with open(path, "rb") as f:
        head = f.read(4096)
    line_end = "\r\n" if b"\r\n" in head else "\n"

    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator=line_end)
        writer.writerow(header)
        writer.writerows(rows)

    if backup:
        bak = path + ".bak"
        if not os.path.exists(bak):
            os.replace(path, bak)
        else:
            os.remove(path)
    # os.replace(tmp_path, path)
    # print("      -> written")
    # return True
    os.replace(tmp_path, path)
    print("      -> written")

    if backup:
        verify_file(path + ".bak", path, column, marker)
    else:
        print("      (no --backup, so no verification)")

    return True




def verify_file(old_path, new_path, column, marker, max_report=5):
    """
    Compare the backup against the converted file. READ ONLY - both files are
    opened with mode "r" and nothing is ever written back.

    Deliberately shares no logic with to_relative: it re-reads both files from
    disk and checks the result a different way, so a bug in the conversion
    cannot pass its own exam.

    Prints PASS or the first `max_report` problems. Returns True if all checks pass.
    """
    def read_rows(p):
        # csv.reader returns every cell as a plain string, exactly as written.
        # Numbers are never parsed, so "0.750" stays "0.750" and compares exactly.
        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            return list(csv.reader(f))

    old_rows = read_rows(old_path)
    new_rows = read_rows(new_path)
    problems = []

    # --- structure -------------------------------------------------------
    if not old_rows or not new_rows:
        problems.append("one of the files is empty")
    else:
        if old_rows[0] != new_rows[0]:
            problems.append(f"header changed: {old_rows[0]} -> {new_rows[0]}")
        if len(old_rows) != len(new_rows):
            problems.append(f"row count changed: {len(old_rows)} -> {len(new_rows)}")

    if problems:
        print(f"      VERIFY FAILED: {len(problems)} problem(s)")
        for p in problems:
            print(f"        - {p}")
        return False

    header = new_rows[0]
    col_idx = header.index(column)

    # --- row by row ------------------------------------------------------
    for n, (old_row, new_row) in enumerate(zip(old_rows[1:], new_rows[1:]), start=2):
        if len(old_row) != len(new_row):
            problems.append(f"line {n}: number of cells changed")
            continue

        # 1. every cell that is NOT the path column must be identical
        for i, (a, b) in enumerate(zip(old_row, new_row)):
            if i != col_idx and a != b:
                problems.append(
                    f"line {n}: column '{header[i]}' changed: {a!r} -> {b!r}")

        if col_idx >= len(new_row):
            continue

        old_value = old_row[col_idx]
        new_value = new_row[col_idx]

        if new_value.strip() == "":
            continue                      # blank cell, nothing to check

        old_fwd = old_value.replace("\\", "/")

        # 2. the new path must start at the marker folder
        if not new_value.startswith(marker + "/"):
            problems.append(f"line {n}: does not start with '{marker}/': {new_value!r}")

        # 3. the new path must be the tail of the old one, cut at a folder boundary
        elif not (old_fwd.endswith("/" + new_value) or old_fwd == new_value):
            problems.append(f"line {n}: not a tail of the original: {old_value!r} -> {new_value!r}")

        # 4. no stray whitespace
        if new_value != new_value.strip():
            problems.append(f"line {n}: leading/trailing whitespace: {new_value!r}")

        # 5. no backslashes left
        if "\\" in new_value:
            problems.append(f"line {n}: still contains a backslash: {new_value!r}")

    # --- report ----------------------------------------------------------
    if not problems:
        print(f"      VERIFY OK  ({len(new_rows) - 1} rows checked)")
        return True

    print(f"      VERIFY FAILED: {len(problems)} problem(s)")
    for p in problems[:max_report]:
        print(f"        - {p}")
    if len(problems) > max_report:
        print(f"        ... and {len(problems) - max_report} more")
    return False



# arguments 
parser = argparse.ArgumentParser(
    description="Make the file_path column of CSVs relative."
)
parser.add_argument("--folder", required=True,
                    help="folder containing the CSV files")
parser.add_argument("--recursive", action="store_true",
                    help="also process subfolders")
parser.add_argument("--apply", action="store_true",
                    help="write the changes (without this it is a dry run)")
parser.add_argument("--backup", action="store_true",
                    help="keep the original as <name>.csv.bak")
parser.add_argument("--column", default="file_path",
                    help="column to convert (default: file_path)")
parser.add_argument("--marker", default="data",
                    help="folder to start the relative path from (default: data)")
args = parser.parse_args()

if not os.path.isdir(args.folder):
    sys.exit(f"Not a folder: {args.folder}")

# collect CSVs paths into a files list to iterate over

files = []
if args.recursive:
    for root, _dirs, names in os.walk(args.folder):
        for name in sorted(names):
            if name.lower().endswith(".csv"):
                files.append(os.path.join(root, name))
else:
    for name in sorted(os.listdir(args.folder)):
        full = os.path.join(args.folder, name)
        if name.lower().endswith(".csv") and os.path.isfile(full):
            files.append(full)

if not files:
    sys.exit(f"No CSV files found in {args.folder}")

# run it

mode = "APPLY (files will be written)" if args.apply else "DRY RUN (nothing is written)"
print(mode)
print(f"Folder: {args.folder}   CSVs found: {len(files)}\n")

changed = 0
for path in files:
    changed += bool(
        process_file(path, args.column, args.marker, args.apply, args.backup)
    )

print()
if args.apply:
    print(f"Done. {changed} file(s) changed.")
else:
    print(f"Done. {changed} file(s) would change. Re-run with --apply to write.")