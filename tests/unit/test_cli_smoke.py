from typer.testing import CliRunner

from rfmna.cli.main import app

runner = CliRunner()


def test_cli_help_smoke() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "check" in result.stdout
    assert "run" in result.stdout
