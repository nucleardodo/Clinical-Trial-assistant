"""Recommendation engine."""

def recommendations(current_health:str,target:str):
 cur=current_health.lower();tar=target.lower();acts=[]
 if cur=="poor" and tar=="good":
  acts=["Expand pre-screening","Increase referral outreach","Review restrictive criteria","Weekly recruitment review"]
 elif cur=="good" and tar=="excellent":
  acts=["Increase investigator engagement","Optimize patient retention","Improve conversion rate","Add contingency recruitment"]
 else:
  acts=["Monitor KPIs","Review protocol impact"]
 return acts
