import typer

app = typer.Typer(help="RF MNA solver CLI")

@app.command()
def check(design: str) -> None:
    """Preflight checks for a design file."""
    raise typer.Exit(code=0)

@app.command()
def run(design: str, analysis: str = "ac") -> None:
    """Run simulation."""
    if analysis != "ac":
        raise typer.BadParameter("Only AC is supported in v4.")
    raise typer.Exit(code=0)
