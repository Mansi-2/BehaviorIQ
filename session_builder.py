import pandas as pd
from config import DOPAMINE_APPS

def build_sessions(df):
    sessions = []
    current_app = None
    start_time = None

    for _, row in df.iterrows():
        if row["event_type"] == "ACTIVITY_RESUMED":
            current_app = row["package"]
            start_time = row["timestamp"]

        elif row["event_type"] == "ACTIVITY_PAUSED" and current_app:
            duration = (row["timestamp"] - start_time).total_seconds()
            sessions.append({
                "app": current_app,
                "start": start_time,
                "duration": duration
            })
            current_app = None

    sessions = pd.DataFrame(sessions)

    if len(sessions) > 0:
        sessions["hour"] = sessions["start"].dt.hour
        sessions["weekday"] = sessions["start"].dt.day_name()
        sessions["date"] = sessions["start"].dt.date
        sessions["month"] = sessions["start"].dt.to_period("M")
        sessions["dopamine"] = sessions["app"].isin(DOPAMINE_APPS)

    return sessions