"""
Logging utilities for consistent formatting across visualization scripts.
Uses rich library if available, otherwise falls back to simple formatting.
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


def print_step(current: int, total: int, description: str):
    """
    Print a step progress indicator.
    Uses rich formatting if available, otherwise falls back to simple format.

    Args:
        current: Current step number (1-indexed)
        total: Total number of steps
        description: Description of the current step

    Example:
        print_step(1, 5, "Ecosystem diagnostics (TChl, EXP, PPINT)")
        # With rich:
        #   [1/5] Ecosystem diagnostics (TChl, EXP, PPINT)...  (in cyan)
        # Without rich:
        #   [1/5] Ecosystem diagnostics (TChl, EXP, PPINT)...
    """
    message = f"  [{current}/{total}] {description}..."

    if HAS_RICH:
        _console.print(message, style="cyan")
    else:
        print(message)


def print_success(message: str):
    """
    Print a success message.
    Uses rich formatting if available, otherwise falls back to simple format.

    Args:
        message: Success message to display

    Example:
        print_success("Spatial maps complete")
        # With rich:
        #   ✓ Spatial maps complete  (in green)
        # Without rich:
        #   ✓ Spatial maps complete
    """
    full_message = f"✓ {message}"

    if HAS_RICH:
        _console.print(full_message, style="green")
    else:
        print(full_message)
