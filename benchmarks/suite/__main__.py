"""
CLI entry point: python -m benchmarks.suite

Usage:
    python -m benchmarks.suite run --mode default
    python -m benchmarks.suite run --mode hpo
    python -m benchmarks.suite run --mode all
    python -m benchmarks.suite run --mode default --category reg_low_d_smooth
    python -m benchmarks.suite run --mode default --dataset diabetes
    python -m benchmarks.suite run --mode default --save-models
    python -m benchmarks.suite list
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="benchmarks.suite",
        description="GeoXGB vs XGBoost benchmarking suite",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_p = sub.add_parser("run", help="Run benchmarks")
    run_p.add_argument("--mode", choices=["default", "hpo", "all"],
                       default="default",
                       help="Benchmark mode (default: default)")
    run_p.add_argument("--category", type=str, default=None,
                       help="Filter to a single category")
    run_p.add_argument("--dataset", type=str, default=None,
                       help="Filter to a single dataset name")
    run_p.add_argument("--save-models", action="store_true",
                       help="Serialize fitted models for later inspection")
    run_p.add_argument("--output", type=str, default=None,
                       help="Output CSV path (default: results/<mode>_results.csv)")
    run_p.add_argument("--workers", type=int, default=1,
                       help="Parallel dataset workers (default: 1 = sequential)")

    # --- list ---
    list_p = sub.add_parser("list", help="List available datasets and categories")
    list_p.add_argument("--category", type=str, default=None,
                        help="Filter to a single category")

    args = parser.parse_args()

    if args.command == "list":
        from .datasets import get_datasets, list_categories
        if args.category:
            datasets = get_datasets(category=args.category)
        else:
            datasets = get_datasets()
        categories = list_categories()
        print(f"\n{len(categories)} categories, {len(datasets)} datasets total\n")
        current_cat = None
        for ds in sorted(datasets, key=lambda d: (d["category"], d["name"])):
            if ds["category"] != current_cat:
                current_cat = ds["category"]
                print(f"\n[{current_cat}]")
            print(f"  {ds['name']:30s}  task={ds['task']}")
        print()

    elif args.command == "run":
        from .runner import run
        run(
            mode=args.mode,
            category=args.category,
            dataset_name=args.dataset,
            save_models=args.save_models,
            output=args.output,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
