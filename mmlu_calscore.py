import fire
import pandas as pd


def main(file_name: str):
    df = pd.read_csv(file_name)
    print("\n### All Average ###")
    print(df.groupby("adapter_name").aggregate({"acc_score": "mean"}))
    print("\n### Category Average ###")
    print(
        df.groupby(["adapter_name", "mmlu_categories"]).aggregate({"acc_score": "mean"})
    )


if __name__ == "__main__":
    fire.Fire(main)
