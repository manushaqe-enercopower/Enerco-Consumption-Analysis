import subprocess
import sys

PIPELINES = [
    "src.quality",
    "src.reshape",
    "src.metrics",
    "src.outliers",
    "src.weather",
    "src.holidays",
    "src.tariff",
    "src.clustering",
    "src.prosumers",
]


def main():
    for pipeline in PIPELINES:
        print(f"\n{'=' * 70}")
        print(f"Running: {pipeline}")
        print("=" * 70)

        result = subprocess.run(
            [sys.executable, "-m", pipeline],
            check=False,
        )

        if result.returncode != 0:
            print(f"\nPipeline failed: {pipeline}")
            sys.exit(result.returncode)

    print("\n" + "=" * 70)
    print("ALL PIPELINES COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
