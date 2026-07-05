"""Session context helpers."""

def get_context(state):
    return state.setdefault("advisor_context",{})

def update_context(state, **kwargs):
    ctx=get_context(state)
    ctx.update(kwargs)
    return ctx

def last_site(state):
    return get_context(state).get("site")
