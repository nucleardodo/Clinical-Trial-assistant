"""
site_metrics.py
Clinical Trial Site Performance Metrics Engine v1.0
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd
from config import HEALTH_THRESHOLDS

@dataclass
class SiteRecord:
    site_id:str
    site_name:str
    health:str
    data:Dict[str,Any]

class SiteMetrics:
    def __init__(self,dataframe:pd.DataFrame):
        self.df=dataframe.copy()
        self.metrics_df=None
        self.validate_dataframe()
        self.normalize_dataframe()

    def validate_dataframe(self):
        if self.df.empty:
            raise ValueError("Dataframe is empty.")
        req=["site_id","site_name","site_health"]
        miss=[c for c in req if c not in self.df.columns]
        if miss:
            raise ValueError(f"Missing required columns: {miss}")

    def normalize_dataframe(self):
        self.df.columns=[c.strip().lower() for c in self.df.columns]
        for c in self.df.select_dtypes(include="object").columns:
            self.df[c]=self.df[c].fillna("").astype(str).str.strip()

    @staticmethod
    def classify_health(score:float)->str:
        if score>=HEALTH_THRESHOLDS["Excellent"]: return "Excellent"
        if score>=HEALTH_THRESHOLDS["Good"]: return "Good"
        if score>=HEALTH_THRESHOLDS["Fair"]: return "Fair"
        return "Poor"

    @staticmethod
    def classify_screen_failure(v):
        try:v=float(v)
        except:return "Unknown"
        return "Low" if v<=0.10 else "Moderate" if v<=0.20 else "High"

    @staticmethod
    def classify_timeline_risk(v):
        try:v=float(v)
        except:return "Unknown"
        return "Low" if v<=5 else "Medium" if v<=15 else "High"

    def get_site(self,identifier:str)->Optional[SiteRecord]:
        s=identifier.lower()
        for _,r in self.df.iterrows():
            if s==r["site_id"].lower() or s in r["site_name"].lower():
                return SiteRecord(r["site_id"],r["site_name"],r["site_health"],r.to_dict())
        return None

    def calculate_metrics(self):
        df=self.df.copy()
        if {"current_enrollment","target_enrollment"}<=set(df.columns):
            df["enrollment_efficiency"]=(df["current_enrollment"]/df["target_enrollment"]).fillna(0).clip(upper=1)
            df["recruitment_gap"]=df["target_enrollment"]-df["current_enrollment"]
        else:
            df["enrollment_efficiency"]=0
            df["recruitment_gap"]=0
        if "screen_failure_rate" in df.columns:
            df["screen_failure_severity"]=df["screen_failure_rate"].apply(self.classify_screen_failure)
        else: df["screen_failure_severity"]="Unknown"
        if "timeline_delay_days" in df.columns:
            df["timeline_risk"]=df["timeline_delay_days"].apply(self.classify_timeline_risk)
        else: df["timeline_risk"]="Unknown"
        scores=[]
        for _,r in df.iterrows():
            sc=100+r["enrollment_efficiency"]*25
            if "screen_failure_rate" in df.columns: sc-=r["screen_failure_rate"]*30
            if "timeline_delay_days" in df.columns: sc-=r["timeline_delay_days"]*0.75
            scores.append(round(max(0,min(100,sc)),1))
        df["performance_score"]=scores
        df["calculated_health"]=df["performance_score"].apply(self.classify_health)
        self.metrics_df=df
        return df

    def get_ai_metrics(self,identifier):
        if self.metrics_df is None: self.calculate_metrics()
        s=self.get_site(identifier)
        if not s: return None
        r=self.metrics_df[self.metrics_df.site_id==s.site_id].iloc[0]
        return {"site_id":r.site_id,"site_name":r.site_name,"health":r.calculated_health,
                "performance_score":r.performance_score,
                "enrollment_efficiency":r.enrollment_efficiency,
                "recruitment_gap":r.recruitment_gap,
                "screen_failure":r.screen_failure_severity,
                "timeline_risk":r.timeline_risk,
                "raw_data":r.to_dict()}

    def rank_sites(self,ascending=False):
        if self.metrics_df is None: self.calculate_metrics()
        return self.metrics_df.sort_values("performance_score",ascending=ascending).reset_index(drop=True)

    def get_best_site(self):
        return self.rank_sites().iloc[0]

    def get_highest_risk_site(self):
        return self.rank_sites(ascending=True).iloc[0]

    def get_portfolio_summary(self):
        if self.metrics_df is None: self.calculate_metrics()
        df=self.metrics_df
        return {"total_sites":len(df),
                "average_performance":round(df.performance_score.mean(),2),
                "average_enrollment_efficiency":round(df.enrollment_efficiency.mean(),2),
                "portfolio_health":df.calculated_health.value_counts().to_dict()}

    def compare_sites(self,a,b):
        s1=self.get_ai_metrics(a); s2=self.get_ai_metrics(b)
        if not s1 or not s2: return None
        return {"site_1":s1,"site_2":s2,
                "winner":s1["site_name"] if s1["performance_score"]>=s2["performance_score"] else s2["site_name"],
                "score_difference":abs(s1["performance_score"]-s2["performance_score"])}

    def get_dashboard_metrics(self):
        return self.get_portfolio_summary()
