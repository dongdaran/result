# README 실행 명령의 HF 환경변수 접두어 제거

- 날짜: 2026-10-13 (UTC)
- session id: `01a13446-d303-7f72-ab96-9719cc0287d3`
- [작업 목록](../overview.md)
- [이전 의존성 복원 작업](outputs_github-설정-복원.md)

## 요청과 변경

README의 명령을 환경변수 접두어 없이 실행할 수 있도록 수정했다.

- `pyproject.toml`: GPU extra에 `hf-transfer>=0.1.9,<0.2` 추가.
- `uv.lock`: hf-transfer 0.1.9 추가. PyTorch 2.14.1+cu130과 기존 CUDA 13 의존성 유지.
- `README.md`: `HF_HUB_ENABLE_HF_TRANSFER=0` 접두어 제거 및 의존성 안내 수정.
- 이전 로그는 당시 수행 기록으로 보존. 현재 실행 방법은 README와 이 로그를 따른다.

환경변수 자체를 삭제하거나 전송 설정을 강제로 변경하지 않았다. Runpod에서 이미 활성화된 HF 전송 옵션에 필요한 패키지를 설치해 일반 명령을 지원한다.

새 잠금 파일 SHA-256: `c68d0a6a6a182181390e53fdd59869e164530f197220cbc9ea99c8897b68605b`

의존성을 추가했으므로 잠금 파일 hash는 outputs_github와 더 이상 같지 않다. PyTorch/CUDA 버전은 유지했다. 기존 산출물 snapshot은 수정하지 않았다.

## 검증

| 확인 | 결과 |
|---|---|
| `uv lock` | hf-transfer 0.1.9만 추가 |
| `uv sync --frozen --extra gpu --extra dev` | 설치 성공 |
| `uv run --extra gpu --extra dev ruff check .` | 통과 |
| 임시 빈 캐시에서 지정 revision의 config.json 다운로드 | 성공. HF 전송 옵션 활성화(True), 추가 환경변수 접두어 없음 |
| `./scripts/preflight --model model_1` | 모델 config·tokenizer 조회 성공. CUDA 사용 불가로 종료 코드 2 |

CUDA 실패는 기존 드라이버 570.195.03(CUDA 12.8)과 PyTorch CUDA 13 빌드의 호환성 문제다. HF 전송 의존성 오류는 해결됐다. CUDA 호환 환경 없이 GPU smoke·추출·생성은 실행하지 않았다. 의존성 추가와 문서 변경만 수행했으며 전체 pytest는 재실행하지 않았다.

## 실행 방법

```sh
uv sync --frozen --extra gpu --extra dev
./scripts/preflight --model model_1
```

CUDA 13 호환 환경에서 preflight가 통과한 뒤 README의 smoke·실험 명령을 실행한다. 이전 run snapshot과 새 의존성을 혼합하지 않도록 새 run ID를 사용한다.
