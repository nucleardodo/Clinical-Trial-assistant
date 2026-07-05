"""Prompt library for Clinical Trial AI."""
PROMPTS={
"explain_poor":"You are a Senior Director of Clinical Operations. Explain why the site is POOR using 5 Whys, business impact, timeline impact, risks and actions.",
"explain_fair":"Explain why the site is FAIR, current bottlenecks and improvements.",
"explain_good":"Explain why the site is GOOD, strengths and what prevents EXCELLENT.",
"explain_excellent":"Explain why the site is EXCELLENT, best practices and sustainability.",
"improve_poor":"Provide prioritized plan to move POOR to GOOD with expected gains.",
"improve_good":"Provide prioritized plan to move GOOD to EXCELLENT with expected gains.",
"compare":"Compare two sites objectively.",
"executive":"Produce executive portfolio summary.",
"timeline":"Explain timeline slippage drivers.",
"screen_failure":"Explain screen failure drivers.",
"recruitment":"Explain recruitment challenges.",
"budget":"Recommend budget allocation."
}

def get_prompt(name): return PROMPTS.get(name,PROMPTS["executive"])
