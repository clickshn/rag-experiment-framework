---
name: unit-check
description: 린트·포맷·타입·시험·pre-commit(gitleaks 포함)을 모두 돌려 결과를 표로 요약한다. 단위 구현 뒤, 커밋 전에 쓴다.
disable-model-invocation: true
---

# /unit-check

**보고는 한국어로 쓴다.** 명령 출력이 영어여도 결과 표·요약·설명은 한국어로 쓴다.

아래 명령을 순서대로 **모두** 돌린다. 하나가 실패해도 나머지를 마저 돌린다.

| # | 검사 | 명령 |
| --- | --- | --- |
| 1 | 잠금 파일 | `uv lock --check` |
| 2 | 린트 | `uv run ruff check` |
| 3 | 포맷 | `uv run ruff format --check` |
| 4 | 타입 | `uv run mypy` |
| 5 | 시험 | `uv run pytest -q` |
| 6 | 커밋 전 검사 | `uv run pre-commit run --all-files` |

결과는 아래 형식으로 보고한다.

| 검사 | 결과 | 요약 |
| --- | --- | --- |
| 린트 | 통과 / 실패 | 실패면 첫 오류 몇 줄 |
| 시험 | 통과 / 실패 | `N passed`, 실패한 시험 이름 |
| … | | |

- 실패가 있으면 원인을 짧게 적고, 고칠지 묻는다. 시험을 끄거나 건너뛰게 바꿔 통과시키지 않는다.
- pre-commit이 파일을 고쳤으면(포맷·줄 끝) 고친 파일 목록을 알린다.
