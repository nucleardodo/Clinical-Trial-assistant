"""Site selector."""
import re

def detect_site(query,df):
 q=query.lower();m=re.search(r"(site[- ]?\\d{3})",q,re.I)
 if m:
  sid=m.group(1).upper().replace(" ","-")
  if sid.startswith("SITE") and "-" not in sid: sid=sid.replace("SITE","SITE-")
  r=df[df["site_id"].str.upper()==sid]
  if not r.empty:return r.iloc[0].to_dict()
 for _,row in df.iterrows():
  if row["site_name"].lower() in q or row["site_name"].split()[0].lower() in q:return row.to_dict()
 return None
