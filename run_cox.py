import argparse
import os
import pandas as pd
import a as core


def run(endpoint: str, splits: int):
    results, summary = core.cox_multiple_random_splits(core.df, splits, endpoint=endpoint)
    return results, summary


def main():
    parser = argparse.ArgumentParser(description="Run CoxPH evaluation (sex-agnostic balanced and sex-specific) with multiple splits and export to Excel.")
    parser.add_argument("--splits", type=int, default=50, help="Number of random 70/30 splits (default: 50)")
    parser.add_argument("--endpoint", type=str, default="both", choices=["PE", "SE", "both"], help="Endpoint to evaluate (PE, SE, or both)")
    parser.add_argument("--out", type=str, default="/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff/results_cox.xlsx", help="Output Excel path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.endpoint == "both":
        _, summary_pe = run("PE", args.splits)
        _, summary_se = run("SE", args.splits)
        with pd.ExcelWriter(args.out) as writer:
            summary_pe.to_excel(writer, sheet_name='PE', index=True, index_label='RowName')
            summary_se.to_excel(writer, sheet_name='SE', index=True, index_label='RowName')
        print(f"Wrote PE and SE summaries to {args.out}")
    else:
        _, summary = run(args.endpoint, args.splits)
        with pd.ExcelWriter(args.out) as writer:
            summary.to_excel(writer, sheet_name=args.endpoint, index=True, index_label='RowName')
        print(f"Wrote {args.endpoint} summary to {args.out}")


if __name__ == "__main__":
    main()