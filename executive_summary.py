"""Executive summary."""
import pandas as pd

def summarize(df:pd.DataFrame):
 best=df.sort_values("predicted_baseline_rate",ascending=False).iloc[0]
 worst=df.sort_values("predicted_baseline_rate").iloc[0]
 return {"best_site":best["site_name"],"highest_risk":worst["site_name"],"portfolio_health":df["site_health"].value_counts().to_dict()}
