"""examples/ 아래 파일에 실제 서버 주소가 들어가지 않았는지 검사한다(설계 §9.3).

허용: localhost, 127.0.0.1, 0.0.0.0, *.law.go.kr(공개 법령 API), ${VAR} 치환 자리.
그 밖의 http(s) 주소가 있으면 위치를 출력하고 1로 끝낸다.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

URL_RE = re.compile(r"https?://([^/\s\"'<>)\]}]+)", re.IGNORECASE)
ALLOWED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0"}
ALLOWED_SUFFIXES = (".law.go.kr",)


def is_allowed(host: str) -> bool:
    """주소의 호스트 부분이 허용 목록에 드는지 본다."""
    if host.startswith("${"):
        return True
    name = host.rsplit("@", 1)[-1].split(":", 1)[0].lower()
    if name in ALLOWED_HOSTS or name == "law.go.kr":
        return True
    return name.endswith(ALLOWED_SUFFIXES)


def find_violations(path: Path) -> list[tuple[int, str]]:
    """파일에서 허용되지 않는 주소를 (줄 번호, 주소)로 돌려준다."""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    found: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for m in URL_RE.finditer(line):
            if not is_allowed(m.group(1)):
                found.append((lineno, m.group(0)))
    return found


def main(argv: list[str]) -> int:
    failed = False
    for arg in argv:
        path = Path(arg)
        for lineno, url in find_violations(path):
            print(f"{path}:{lineno}: 예제에 실제 서버 주소가 있습니다: {url}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
