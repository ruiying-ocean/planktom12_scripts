"""
Logging utilities for consistent formatting across visualization scripts.
Uses rich library if available, otherwise falls back to simple box drawing.
"""

# Try to import rich for enhanced formatting
try:
    from rich.console import Console
    from rich.panel import Panel
    HAS_RICH = True
    _console = Console()
except ImportError:
    HAS_RICH = False
    _console = None


def print_header(text: str, width: int = 63):
    """
    Print a boxed header with consistent formatting.
    Uses rich Panel if available, otherwise falls back to Unicode box drawing.

    Args:
        text: The text to display in the header
        width: Total width of the box (default 63, ignored when using rich)

    Example:
        print_header("Step 1: Data Preprocessing")
        # With rich:
        # ╭───────────────────────────────────────╮
        # │ Step 1: Data Preprocessing            │
        # ╰───────────────────────────────────────╯
        # Without rich:
        # ╔═══════════════════════════════════════════════════════════════╗
        # ║  Step 1: Data Preprocessing                                  ║
        # ╚═══════════════════════════════════════════════════════════════╝
    """
    if HAS_RICH:
        # Use rich Panel for prettier output
        panel = Panel(text, expand=False, border_style="blue")
        _console.print(panel)
    else:
        # Fallback to simple Unicode box drawing
        content_width = width - 4  # Account for "║  " and "  ║"
        padded_text = f"  {text}".ljust(content_width)

        top = "╔" + "═" * (width - 2) + "╗"
        middle = f"║{padded_text}║"
        bottom = "╚" + "═" * (width - 2) + "╝"

        print(top)
        print(middle)
        print(bottom)
