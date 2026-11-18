# rag-experiment-framework

RAG를 이루는 부품(생성 모델, 청킹, 임베딩, 벡터DB, 검색 전략, 리랭크, 생성 전략, 판정 모델)을 바꿔 끼우며, **다시 만들 수 있고 공정하게** 비교하는 실험 프레임워크입니다. 구성·채점·문항 세트를 해시로 버전화하고, 평가 원칙(동결, 봉인, 블라인드 검토)을 기본 동작으로 둡니다. 상용 API를 쓸 수 없는 폐쇄망 환경에서 로컬 모델과 로컬 검색 부품만으로 충분한지 검증하려는 팀을 주로 생각하고 만듭니다.

## 상태

**pre-alpha**. 첫 단계인 R0 최소판("어떤 모델을 쓸까")을 만들고 있습니다. 지금은 구성 검사(`ragexp config check`)와 법령 말뭉치 받기(`ragexp corpus fetch-law`)만 쓸 수 있습니다.

- 설계: [docs/design.md](docs/design.md)
- R0 단위 계획: [docs/units/R0.md](docs/units/R0.md)

## 개발 시작

[uv](https://docs.astral.sh/uv/)가 필요합니다.

```bash
uv sync                          # 가상환경과 의존성 설치
uv run pre-commit install        # 커밋 전 검사(ruff, gitleaks 등) 켜기
uv run ragexp --help
```

커밋 전 비밀값 검사에는 [gitleaks](https://github.com/gitleaks/gitleaks)가 PATH에 있어야 합니다.

검사 명령:

```bash
uv run pytest                    # 시험(GPU·네트워크 없이 돈다)
uv run ruff check                # 린트
uv run ruff format --check       # 포맷
uv run mypy                      # 타입 검사
uv run pre-commit run --all-files
```

API 주소와 키는 환경 변수로만 넘깁니다. 변수 목록은 [.env.example](.env.example)에 있습니다.

## 법령 말뭉치 받기

국가법령정보센터 Open API에서 법령의 기준일 판(그날 시행 중인 판)을 받아 스냅샷을 만듭니다. 인증키(OC)는 공동활용 승인과 법령 종류 신청이 끝나 있어야 합니다.

```bash
uv run ragexp corpus fetch-law --law "국가재정법" --law "개인정보 보호법" --as-of 2026-10-29
```

- 결과는 `workspace/snapshots/law-ko-<기준일>/`에 생깁니다(`--out`으로 바꿈). 법령마다 본문 XML, `manifest.json`(파일별 해시·기준일·받은 날짜), `SOURCE.md`(출처표시)가 들어갑니다. 스냅샷은 한 번 만들면 바꾸지 않습니다.
- 프록시는 기본으로 쓰지 않습니다. 회사 프록시가 필요하면 `--proxy env`를 붙입니다.
- 출처: 국가법령정보센터(법제처), 공공누리 제1유형. 법적 효력이 없는 참고자료입니다.

**인증키(OC)를 넣는 법**:
- 환경 변수 `LAW_API_OC`로만 받습니다. 명령줄 옵션은 없습니다.
- Claude Code 같은 도구 안에서 돌린다면, 도구를 띄우기 **전에** OS나 셸의 환경 변수로 넣으세요. 도구의 대화 줄(`!` 명령 등)에 값을 쓰면 대화 기록에 남습니다.
- ragexp는 `.env` 파일을 자동으로 읽지 않습니다.
- PowerShell에서 화면에 보이지 않게 넣으려면 `Read-Host -AsSecureString`의 결과를 평문으로 바꿔 넣어야 합니다. 그대로 넣으면 값이 `System.Security.SecureString`이라는 글자가 됩니다.

  ```powershell
  $s = Read-Host -AsSecureString
  $env:LAW_API_OC = [Net.NetworkCredential]::new('', $s).Password
  ```

## 라이선스

[Apache License 2.0](LICENSE). Copyright 2026 석한나.

---

## English summary

`rag-experiment-framework` (`ragexp`) is a framework for comparing RAG components (generator models, chunking, embeddings, vector stores, retrieval, reranking, prompting, and judge models) in a reproducible and fair way. Configurations, scoring, and question sets are versioned by hash. Evaluation safeguards such as freezing, sealed test sets, and blind human review are on by default. The project is pre-alpha, and the docs are written mainly in Korean. `ragexp corpus fetch-law` builds a snapshot of Korean statutes from the Korea Law Information Center Open API; the API key is read only from the `LAW_API_OC` environment variable. Licensed under Apache-2.0.
