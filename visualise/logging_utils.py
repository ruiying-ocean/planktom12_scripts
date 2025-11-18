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
        # Calculate content width: total width minus 2 for box chars (║)
        content_width = width - 2  # Just the inner width between the box characters
        # Center the text with padding on both sides
        padded_text = text.center(content_width)

        top = "╔" + "═" * content_width + "╗"
        middle = f"║{padded_text}║"
        bottom = "╚" + "═" * content_width + "╝"

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
        #   [1/5] Ecosystem diagnostics (TChl, EXP, PPINT)...  (formatted with colors)
        # Without rich:
        #   [1/5] Ecosystem diagnostics (TChl, EXP, PPINT)...
    """
    if HAS_RICH:
        # Use rich markup for better formatting
        _console.print(f"  [bold cyan]\\[{current}/{total}][/bold cyan] {description}...")
    else:
        print(f"  [{current}/{total}] {description}...")


def print_success(message: str):
    """
    Print a success message.
    Uses rich formatting if available, otherwise falls back to simple format.

    Args:
        message: Success message to display

    Example:
        print_success("Spatial maps complete")
        # With rich:
        #   ✓ Spatial maps complete  (in green, bold checkmark)
        # Without rich:
        #   ✓ Spatial maps complete
    """
    if HAS_RICH:
        # Use rich markup - green text with bold checkmark
        _console.print(f"[bold green]✓[/bold green] [green]{message}[/green]")
    else:
        # Fallback to simple Unicode checkmark
        print(f"✓ {message}")


def print_info(message: str):
    """
    Print an informational message.
    Uses rich formatting if available, otherwise falls back to simple format.

    Args:
        message: Info message to display

    Example:
        print_info("Loading data from /path/to/file")
        # With rich:
        #   → Loading data...  (in blue/cyan)
        # Without rich:
        #   Loading data...
    """
    if HAS_RICH:
        _console.print(f"[cyan]→[/cyan] {message}")
    else:
        print(f"  {message}")


def print_warning(message: str):
    """
    Print a warning message.
    Uses rich formatting if available, otherwise falls back to simple format.

    Args:
        message: Warning message to display

    Example:
        print_warning("File not found, using default")
        # With rich:
        #   ! File not found...  (in red, bold exclamation)
        # Without rich:
        #   Warning: File not found...
    """
    if HAS_RICH:
        _console.print(f"[bold red]![/bold red] [red]{message}[/red]")
    else:
        print(f"Warning: {message}")


def print_error(message: str):
    """
    Print an error message.
    Uses rich formatting if available, otherwise falls back to simple format.

    Args:
        message: Error message to display

    Example:
        print_error("Could not load required data file")
        # With rich:
        #   ✗ Could not load required data file  (in red, bold X)
        # Without rich:
        #   Error: Could not load required data file
    """
    if HAS_RICH:
        _console.print(f"[bold red]✗[/bold red] [red]{message}[/red]")
    else:
        print(f"Error: {message}")
