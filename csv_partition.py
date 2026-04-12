import argparse
from pathlib import Path

import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a CSV file into smaller CSV files."
    )
    parser.add_argument("input_csv", help="Path to the input CSV file")
    parser.add_argument(
        "--rows-per-file",
        type=int,
        help="Number of rows to include in each output file",
    )
    parser.add_argument(
        "--percent",
        type=float,
        help="Split by percentage size. Example: 50 creates 2 files, 25 creates 4 files.",
    )
    parser.add_argument(
        "--output-prefix",
        default="split",
        help="Prefix for generated files. Default: split",
    )
    return parser.parse_args()


def validate_args(args):
    if (args.rows_per_file is None) == (args.percent is None):
        raise ValueError("Provide exactly one of --rows-per-file or --percent.")

    if args.rows_per_file is not None and args.rows_per_file <= 0:
        raise ValueError("--rows-per-file must be greater than 0.")

    if args.percent is not None:
        if args.percent <= 0 or args.percent > 100:
            raise ValueError("--percent must be greater than 0 and at most 100.")

        num_files = 100 / args.percent
        if not num_files.is_integer():
            raise ValueError(
                "--percent must divide 100 evenly. For example: 50, 25, 20, 10."
            )


def split_by_rows(df, rows_per_file, output_prefix):
    total_rows = len(df)
    total_files = (total_rows + rows_per_file - 1) // rows_per_file

    for index in range(total_files):
        start = index * rows_per_file
        end = start + rows_per_file
        chunk = df.iloc[start:end]
        output_file = f"{output_prefix}_{index + 1}.csv"
        chunk.to_csv(output_file, index=False)
        print(f"Created {output_file} with {len(chunk)} rows")


def split_by_percent(df, percent, output_prefix):
    total_files = int(100 / percent)
    chunks = [chunk for chunk in np.array_split(df, total_files) if not chunk.empty]

    for index, chunk in enumerate(chunks, start=1):
        output_file = f"{output_prefix}_{index}.csv"
        chunk.to_csv(output_file, index=False)
        print(f"Created {output_file} with {len(chunk)} rows")


def main():
    args = parse_args()
    validate_args(args)

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    if args.percent is not None:
        split_by_percent(df, args.percent, args.output_prefix)
    else:
        split_by_rows(df, args.rows_per_file, args.output_prefix)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
