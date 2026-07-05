"""Standard response formatter."""

def format_response(summary, findings, recommendations, confidence="High"):
    return f"""## Executive Summary\n{summary}\n\n## Operational Findings\n{findings}\n\n## Recommendations\n{recommendations}\n\n## Confidence\n{confidence}\n"""
