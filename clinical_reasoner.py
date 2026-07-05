"""Clinical reasoning engine."""
from dataclasses import dataclass

@dataclass
class SiteAssessment:
    risk_score: float
    confidence: float
    bottlenecks:list
    strengths:list


def assess_site(site:dict):
    base=float(site.get("baseline_enrollment_rate",0))
    pred=float(site.get("predicted_baseline_rate",base))
    sf=float(site.get("screen_failure_rate",0))
    eff=(pred/base) if base else 0
    risk=round((1-eff)*60+sf*40,1)
    strengths=[]; bottlenecks=[]
    if eff>=0.8: strengths.append("Strong enrollment efficiency")
    else: bottlenecks.append("Enrollment velocity below target")
    if sf<0.2: strengths.append("Low screen failure")
    else: bottlenecks.append("High screen failure")
    if risk<30: status="Excellent"
    elif risk<50: status="Good"
    elif risk<70: status="Fair"
    else: status="Poor"
    return {"health":status,"assessment":SiteAssessment(risk,0.9,bottlenecks,strengths)}
