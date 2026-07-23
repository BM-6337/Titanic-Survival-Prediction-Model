import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

DATA_DIR = "data"
OUTPUT_FILE = "submission.csv"


def load_data():
    train_data = pd.read_csv(f"{DATA_DIR}/train.csv")
    test_data = pd.read_csv(f"{DATA_DIR}/test.csv")
    return train_data, test_data


def engineer_features(df):
    """Add a few derived features on top of the raw columns."""
    df = df.copy()

    # Family size and whether the passenger was travelling alone
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Extract title from name (Mr, Mrs, Miss, Master, rare titles grouped)
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]*)\.")
    rare_titles = df["Title"].value_counts()
    rare_titles = rare_titles[rare_titles < 10].index
    df["Title"] = df["Title"].replace(rare_titles, "Rare")

    # Fill missing Age with median per Title group, missing Fare with overall median
    df["Age"] = df.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Fill missing Embarked with the mode
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    return df


def build_features(train_data, test_data):
    train_data = engineer_features(train_data)
    test_data = engineer_features(test_data)

    features = [
        "Pclass", "Sex", "Age", "SibSp", "Parch",
        "Fare", "Embarked", "FamilySize", "IsAlone", "Title",
    ]

    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])

    # Align columns in case a category (e.g. rare Title) is missing in test set
    X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)

    y = train_data["Survived"]
    return X, y, X_test, test_data


def train_and_predict(X, y, X_test):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_split=4,
        random_state=1,
        n_jobs=-1,
    )

    # Quick 5-fold CV score to sanity-check before writing predictions
    scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-val accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    model.fit(X, y)
    predictions = model.predict(X_test)
    return predictions


def main():
    train_data, test_data = load_data()
    X, y, X_test, test_data = build_features(train_data, test_data)
    predictions = train_and_predict(X, y, X_test)

    output = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predictions,
    })
    output.to_csv(OUTPUT_FILE, index=False)
    print(f"Submission saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
