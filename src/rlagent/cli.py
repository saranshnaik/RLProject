"""Command-line entry point. Subcommands are added as each phase lands."""

import typer

app = typer.Typer(
    help="rlagent: train, evaluate, record and compare RL agents.", no_args_is_help=True
)


@app.callback()
def _main() -> None:
    """Root command group."""


if __name__ == "__main__":
    app()
