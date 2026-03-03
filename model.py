import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from session_builder import build_sessions

def train_model(df):

    sessions = build_sessions(df)

    if len(sessions) == 0:
        return None, None, None

    # Convert duration to minutes
    sessions["duration_minutes"] = sessions["duration"] / 60

    # Target variable
    sessions["target"] = sessions["dopamine"].astype(int)

    features = sessions[["hour", "duration_minutes"]]
    target = sessions["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    feature_importance = pd.DataFrame({
        "Feature": features.columns,
        "Coefficient": model.coef_[0]
    })

    return accuracy, cm, feature_importance