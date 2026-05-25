import textwrap
import shutil


def print_header(text: str) -> None:
    """
    Print the given string prepended by a divider for better visibility.
    """
    print(f"=====> {text}")


def print_divider() -> None:
    """
    Print a divider across the entire width of the terminal window.
    """
    width = shutil.get_terminal_size().columns
    print("-" * width)


def preview_text(text: str, margin_lines: int = 5) -> str:
    cols, rows = shutil.get_terminal_size()

    usable_rows = max(rows - margin_lines, 1)
    max_chars = cols * usable_rows

    clipped = text[:max_chars]

    # Wrap nicely so it respects terminal width
    return "\n".join(textwrap.wrap(clipped, width=cols, replace_whitespace=False))
