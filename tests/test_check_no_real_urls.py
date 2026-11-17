"""scripts/check_no_real_urls.py 시험."""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_no_real_urls.py"
_spec = importlib.util.spec_from_file_location("check_no_real_urls", _SCRIPT)
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


@pytest.mark.parametrize(
    "text",
    [
        "base_url: ${LOCAL_URL}",
        "base_url: http://localhost:8000/v1",
        "base_url: http://127.0.0.1:8000/v1",
        "url: https://www.law.go.kr/DRF/lawService.do",
        "url: http://law.go.kr/",
        "base_url: http://${HOST}/v1",
        "주소 없음",
    ],
)
def test_allowed(tmp_path: Path, text: str) -> None:
    f = tmp_path / "ok.yaml"
    f.write_text(text, encoding="utf-8")
    assert mod.main([str(f)]) == 0


@pytest.mark.parametrize(
    "text",
    [
        "base_url: http://10.0.0.5:8000/v1",
        "base_url: https://api.example.com/v1",
        "base_url: https://user:pw@gpu.internal:8443/v1",
        "url: https://evil-law.go.kr.attacker.com/",
    ],
)
def test_rejected(tmp_path: Path, text: str, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "bad.yaml"
    f.write_text(text, encoding="utf-8")
    assert mod.main([str(f)]) == 1
    assert "bad.yaml:1" in capsys.readouterr().out
