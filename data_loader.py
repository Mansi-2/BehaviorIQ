import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.day_name()
    df["date"] = df["timestamp"].dt.date
    df["month"] = df["timestamp"].dt.to_period("M")

    return df