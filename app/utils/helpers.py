import markdown

def render_markdown(text: str) -> str:
    return markdown.markdown(text, extensions=['extra'])