"""Build structured prompts for the LLM."""
from prompt_library import get_prompt

def build_prompt(intent, site, protocol="", metrics=None):
    system=get_prompt(intent)
    metrics=metrics or {}
    user=f"""Selected Site: {site.get('site_name','N/A')}\nSite ID: {site.get('site_id','N/A')}\nHealth: {site.get('site_health','Unknown')}\nMetrics: {metrics}\nProtocol Restrictions:\n{protocol}\nProvide a concise, evidence-based response using only the supplied information."""
    return system,user
