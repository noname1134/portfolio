import pandas as pd


def calc_precision(df):
    return df["actual_churn"].sum() / len(df)


def calc_net_profit(df):
    actual_churners_contacted = df["actual_churn"].sum()
    profit = actual_churners_contacted * 500
    cost = len(df) * 50
    net_profit = profit - cost
    return net_profit


if __name__ == "__main__":
    df = pd.DataFrame({"customer": ["A", "B", "C", "D", "E", "F"],
    "actual_churn": [1, 0, 1, 1, 0, 0],
    "churn_probability": [0.90, 0.80, 0.70, 0.40, 0.20, 0.10]})
    sorted_df = df.sort_values(by=["churn_probability"], ascending=False)
    top_2_df = sorted_df.head(2)
    threshold_df = df.loc[df["churn_probability"] >= 0.5]
    for name, selected in [
        ("Top 2", top_2_df),
        ("Threshold >= 0.5", threshold_df)
    ]:
        print(f"\n{name}")
        print(f"Customers contacted: {len(selected)}")
        print(f"Precision: {calc_precision(selected):.2%}")
        print(f"Net profit: ₪{calc_net_profit(selected)}")
