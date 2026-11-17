**모든 응답·보고는 한국어로 쓴다.** 도구 결과나 원문이 영어여도, 사람에게 하는 답변·진행 보고·최종 보고는 한국어로 쓴다.

# CLAUDE.md

RAG 부품을 바꿔 끼우며 다시 만들 수 있고 공정하게 비교하는 실험 프레임워크다. 배포명은 `rag-experiment-framework`, import·CLI 이름은 `ragexp`다.

## 정본

- 설계의 정본은 `docs/design.md`다. 구현이 설계와 달라져야 하면 **설계 문서부터 고치고** 단위 메모의 "설계와 다른 점"에 적는다.
- 현재 단계 계획은 `docs/units/R0.md`, 단위별 메모는 `docs/units/U<n>.md`에 있다.

## 작업 방식 (레포를 쓰는 누구나 따른다)

단위 하나는 **메모 → 승인 → 구현 → 시험 → 보고 → 커밋** 순서로 진행한다.

**브랜치·병합:** 단위마다 `u<n>-<이름>` 브랜치 → PR → CI 통과 → **squash 병합**(제목 `U<n>: <요약> (#PR번호)`), 병합 뒤 브랜치 삭제. main에 직접 커밋하지 않는다.

1. **메모**: `/unit-plan <n>`으로 `docs/units/U<n>.md`를 쓴다. 칸은 목표·만들 것·파일·완료 기준·시험 목록·설계와 다른 점·결정 필요다. design-reviewer 에이전트의 검토를 받는다.
2. **승인**: 메모를 사람에게 보여 주고 **멈춘다**. 승인 전에는 구현하지 않는다.
3. **구현·시험**: 메모 범위만 만든다. 범위를 넘는 일은 메모에 "남은 일"로 적는다.
4. **검사**: `/unit-check`를 돌린다.
5. **보고**: `/unit-report`로 메모에 "결과" 절(완료 기준별 통과와 근거)을 쓴다.
6. **커밋**: 단위 브랜치(`u<n>-…`)에 커밋한다. `git add`로 먼저 올린 뒤 커밋하고, `git commit -a`/`--all`은 쓰지 않는다(훅이 막음. staged 비밀값 검사가 보지 못하는 변경이 생김).
7. **push·PR**: **반드시 사람에게 먼저 묻는다.** force push는 하지 않는다.
8. **CI 결과**: CI 결과(작업별 통과·실패, run 번호)는 **PR 본문**에 적는다. 단위 메모의 CI 완료 기준에는 `CI: PR #n`만 적는다. 결과를 기록하려고 커밋을 더 해서 CI를 한 번 더 돌리지 않는다.

- 새 런타임 의존성은 묻고 추가하며, 이유를 ADR(`docs/adr/`)로 남긴다.
- 정해야 할 것이 있으면 추측해서 진행하지 말고 메모의 "결정 필요"에 올린다.

## 언어

- 답변·문서·커밋 메시지·주석·docstring은 한국어로 쓴다.
- 식별자(변수·함수·클래스·파일 이름)는 영어로 쓴다.
- README는 한국어 본문 끝에 짧은 영어 요약을 붙인다.
- 커밋 메시지는 `종류: 요약` 형식이다(종류는 docs·build·feat·fix·test·chore·refactor).

## 명령

```bash
uv sync                              # 의존성 설치
uv run pytest                        # 시험(GPU·네트워크 없이)
uv run ruff check --fix              # 린트
uv run ruff format                   # 포맷
uv run mypy                          # 타입 검사(strict, src·tests)
uv run pre-commit run --all-files    # 커밋 전 검사 전체(gitleaks 포함)
uv run ragexp --help                 # CLI
```

## 지켜야 할 것

- **비밀값**: API 키·법령 API 인증키(OC)·실제 서버 주소는 레포·실행 기록·시험 자료 어디에도 쓰지 않는다. 환경 변수로만 받는다. 변수 이름은 `.env.example`에 적고, `.env`는 읽거나 쓰지 않는다(훅이 막음).
- **예제**: `examples/`에는 `${VAR}`·localhost·law.go.kr 주소만 쓴다(pre-commit `no-real-urls`).
- **시험**: 기본 시험은 GPU·네트워크 없이 돌아야 한다. 모델 서버는 가짜 서버로 바꿔 시험한다. 실제 네트워크가 필요한 시험에는 `@pytest.mark.network`를 붙인다(기본 제외).
- **해시**: 구성 해시에 주소·비밀값·모델 meta(가격 등)를 넣지 않는다(설계 §5).
- **실행 기록**: 실행 기록은 파일이 정본이다. SQLite는 다시 만들 수 있는 색인일 뿐이다(설계 §8).
- **DASP**: DASP 레포의 코드는 옮기지 않는다. 원칙과 경험만 가져온다.

## 환경

- Windows이고 Bash 도구는 Git Bash다. 경로는 슬래시로 쓴다.
- 줄바꿈은 LF다(`.gitattributes`). `.gitattributes`는 그대로 둔다.
- Python은 `.python-version`의 3.12로 개발하고, CI에서 3.12·3.14를 시험한다.

## 하네스

- `.claude/settings.json`: 권한 목록과 훅 세 개
  - `.env` 접근 차단
  - 커밋 전 gitleaks(`-a`/`--all` 커밋 거부)
  - .py를 고치면 ruff
- `.claude/skills/`: 명령 `/unit-plan`, `/unit-check`, `/unit-report`
- `.claude/agents/design-reviewer.md`: 설계 원칙 대조 검토(읽기 전용)
