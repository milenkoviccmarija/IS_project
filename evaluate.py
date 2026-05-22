from pathlib import Path

import pandas as pd


RESULTS_PATH = Path("outputs/dog_detections.csv")


def main() -> None:

    if not RESULTS_PATH.exists():
        print("Ne postoji CSV sa rezultatima.")
        return

    df = pd.read_csv(RESULTS_PATH)

    print(df.head())

    print("\nUkupan broj detekcija:")
    print(len(df))

    print("\nProsečan confidence:")
    print(df["confidence"].mean())


if __name__ == "__main__":
    main()