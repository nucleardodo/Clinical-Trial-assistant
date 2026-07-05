"""Intent classifier."""
import re
INTENTS={"explain_poor":[r"\\bpoor\\b"],"explain_fair":[r"\\bfair\\b"],"explain_good":[r"\\bgood\\b"],"explain_excellent":[r"\\bexcellent\\b"],"improve_poor":[r"become good",r"poor.*good"],"improve_good":[r"become excellent",r"good.*excellent"],"compare_sites":[r"compare",r"\\bvs\\b"],"executive_summary":[r"highest risk",r"summary",r"rank"]}
def classify_intent(query:str)->str:
 q=query.lower();
 
 for k,v in INTENTS.items():
  
  if any(re.search(p,q) for p in v): return k
 return "general_analysis"
