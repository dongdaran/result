# outputs_github dependency lock SHA 정확 일치

- 날짜: 2026-10-13 (UTC)
- session id: `01a13446-d303-7f72-ab96-9719cc0287d3`
- [작업 목록](../overview.md)

## 요청과 결과

outputs_github manifest의 dependency_lock_sha256과 현재 uv.lock의 실제 SHA-256을 정확히 맞췄다. hf-transfer를 추가하면서 생긴 잠금 파일 차이를 제거했다.

```text
outputs_github dependency_lock_sha256:
c721bf9fa2acd09333fdb86fa5ba2501cf73ecc892e4b86773f05926a56aea9a
현재 uv.lock SHA-256:
c721bf9fa2acd09333fdb86fa5ba2501cf73ecc892e4b86773f05926a56aea9a
```

## 변경

- `pyproject.toml`, `uv.lock`: Git HEAD의 원본을 읽어 복원. 원본 uv.lock의 hash가 참조 값과 일치함을 먼저 확인했다.
- `.venv`: `uv sync --frozen --extra gpu --extra dev`로 동기화. hf-transfer 0.1.9 제거. PyTorch 2.14.1+cu130, Transformers 4.57.6 유지.
- `scripts/cuda-compat-env`: GPU wrappers에서 `HF_HUB_ENABLE_HF_TRANSFER=0`을 내부 설정해 표준 HF 다운로드 사용. 사용자 명령에 접두어를 요구하지 않는다. 기존 CUDA 13 compat 라이브러리 적용 유지.
- `README.md`: 정확한 lock hash 일치와 내부 HF 다운로드 설정을 현재 동작에 맞춰 설명.

## 검증

| 확인 | 결과 |
|---|---|
| 현재 uv.lock SHA-256 vs 참조 manifest | 정확 일치 |
| `uv lock --check` | 통과 |
| `uv sync --frozen --extra gpu --extra dev` | 설치 동기화 성공 |
| `./scripts/preflight --model model_1` | 종료 코드 0, CUDA true, A40, bf16 지원, Qwen revision 조회 성공 |
| 빈 임시 캐시에서 Qwen config.json 다운로드 | hf-transfer 미설치·전송 옵션 false인 표준 다운로드 성공 |
| `uv run --extra gpu --extra dev ruff check .` | 통과 |
| `sh -n scripts/cuda-compat-env` | 통과 |

이 변경에서는 전체 pytest나 모델 가중치 로딩·벡터 재추출을 다시 실행하지 않았다.

## 기존 산출물과 새 실행

현재 outputs/smoke-model1-001 manifest의 lock hash는 `c68d0a6a6a182181390e53fdd59869e164530f197220cbc9ea99c8897b68605b`다. 이것은 해당 산출물 생성 당시의 provenance이므로 **현재 잠금 파일에 맞춘다고 기존 manifest의 hash를 수정하지 않았다.** 기존 config snapshot·벡터·응답도 보존했다.

새 잠금 파일로 실험을 실행할 때는 새 run ID를 사용해야 한다. 같은 run ID의 snapshot 불일치를 덮어써 우회하지 않는다. dependency lock 일치는 코드 hash·실행 환경·direction 파일 자체의 SHA까지 모두 같다는 의미가 아니다. CUDA compat 패키지는 Python 잠금 파일 밖의 환경 의존성이다.

```sh
uv sync --frozen --extra gpu --extra dev
./scripts/preflight --model model_1
./scripts/smoke --model model_1 --run-id smoke-model1-reference-lock-001
```

마지막 smoke 명령은 안내용이며 이번 작업에서는 실행하지 않았다. 이전 작업 로그는 당시 상태를 나타내는 기록으로 보존한다.
