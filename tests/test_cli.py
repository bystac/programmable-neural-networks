import subprocess
import sys

import pytest

from pnn.calc import main


@pytest.mark.parametrize(
    "argv, expect",
    [
        (["13", "+", "29", "--bits", "8"], "result =     42"),
        (["200", "-", "57"], "result =    143"),
        (["57", "-", "200"], "borrow=1"),
        (["12", "*", "13"], "result =    156"),
        (["12", "x", "13", "--style", "lookahead"], "result =    156"),
        (["7", "<", "9"], "true"),
        (["9", "<", "7"], "false"),
        (["42", "==", "42", "--mode", "sigmoid"], "true"),
    ],
)
def test_cli(capsys, argv, expect):
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert expect in out and "OK" in out and "neurons" in out


def test_cli_export(tmp_path, capsys):
    path = tmp_path / "add.npz"
    assert main(["1", "+", "2", "--export", str(path)]) == 0
    assert path.exists()


def test_cli_rejects_out_of_range():
    with pytest.raises(SystemExit):
        main(["300", "+", "1", "--bits", "8"])


def test_module_entry_point():
    r = subprocess.run([sys.executable, "-m", "pnn.calc", "13", "+", "29"], capture_output=True, text=True)
    assert r.returncode == 0 and "42" in r.stdout
