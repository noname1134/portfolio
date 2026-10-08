import pandas as pd

students = {
    "hours_studied": [2, 5, 3, 8, 7, 1, 6, 4, 9, 2],
    "attendance": [60, 85, 70, 95, 90, 50, 88, 75, 98, 65],
    "previous_score": [55, 72, 64, 91, 85, 45, 78, 69, 94, 58],
    "passed": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
}

if __name__ == "__main__":
    df = pd.DataFrame(students)
    #print(df)
    #print(f"df shape: {df.shape}")
    #print(f"df columns: {df.columns}")
    #print(f"df dtypes: {df.dtypes}")
    #print(f"df describe(): {df.describe()}")
    #print(df['hours_studied'].mean())
    average_value = df.loc[df['passed'] == 1, 'previous_score'].mean()
    #print(average_value)
    X = df.drop("passed", axis=1)
    y = df["passed"]
    #print(X.head())
    #print(X.shape)
    #print(y.head())
    #print(y.shape)