import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

contracts = [
    ["Month-to-month"],
    ["One year"],
    ["Two year"]
]

telco = pd.DataFrame({
    "age": [25, 42, 31, 55],
    "monthly_charges": [40.5, 90.2, 65.3, 110.0],
    "contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
    "internet_service": ["DSL", "Fiber optic", "DSL", "No"],
    "paperless_billing": ["Yes", "No", "Yes", "Yes"],
    "churn": ["No", "Yes", "No", "Yes"]
})

if __name__ == "__main__":
    df = pd.DataFrame(telco)
    X = df.drop("churn", axis=1)
    y = df["churn"]
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(contracts)
    print(encoded)
    print(encoder.get_feature_names_out(["contract"]))
    depth = 5
    #model = DecisionTreeClassifier(random_state=42, max_depth=depth)
