# rag-experiment-framework

RAG를 이루는 부품(생성 모델, 청킹, 임베딩, 벡터DB, 검색 전략, 리랭크, 생성 전략, 판정 모델)을 바꿔 끼우며, **다시 만들 수 있고 공정하게** 비교하는 실험 프레임워크입니다. 구성·채점·문항 세트를 해시로 버전화하고, 평가 원칙(동결, 봉인, 블라인드 검토)을 기본 동작으로 둡니다. 상용 API를 쓸 수 없는 폐쇄망 환경에서 로컬 모델과 로컬 검색 부품만으로 충분한지 검증하려는 팀을 주로 생각하고 만듭니다.

## 상태

**pre-alpha**. 첫 단계인 R0 최소판("어떤 모델을 쓸까")을 만들고 있습니다. 아직 쓸 수 있는 기능은 없습니다.

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

## 라이선스

[Apache License 2.0](LICENSE). Copyright 2026 석한나.

---

## English summary

`rag-experiment-framework` (`ragexp`) is a framework for comparing RAG components (generator models, chunking, embeddings, vector stores, retrieval, reranking, prompting, and judge models) in a reproducible and fair way. Configurations, scoring, and question sets are versioned by hash. Evaluation safeguards such as freezing, sealed test sets, and blind human review are on by default. The project is pre-alpha, and the docs are written mainly in Korean. Licensed under Apache-2.0.
