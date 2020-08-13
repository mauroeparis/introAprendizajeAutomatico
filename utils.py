from IPython.display import display, Markdown


def display_markdown(*args, **kwargs):
    return display(Markdown(*args, **kwargs))
