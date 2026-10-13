# CUDA 13 호환 라이브러리 적용

- 날짜: 2026-10-13 (UTC)
- session id: `01a13446-d303-7f72-ab96-9719cc0287d3`
- [작업 목록](../overview.md)
- [이전 HF 다운로드 설정](README-명령-HF-접두어-제거.md)

## 요청과 결과

현재 A40 및 R570 드라이버를 유지한 채 CUDA 13용 PyTorch를 실행하도록 NVIDIA forward compatibility 라이브러리를 설치했다. **README의 `./scripts/preflight --model model_1` 명령이 환경변수 접두어 없이 종료 코드 0으로 통과했다.** 실제 CUDA float32 행렬곱과 bfloat16 attention 연산도 통과했다.

## 환경과 설치

| 항목 | 값 |
|---|---|
| OS | Ubuntu 24.04.3 LTS |
| GPU | NVIDIA A40 |
| 호스트 NVIDIA 드라이버 | 570.195.03 (변경하지 않음) |
| PyTorch / CUDA runtime | 2.14.1+cu130 / 13.0 (유지) |
| Transformers / Python | 4.57.6 / 3.11.13 |
| 설치 패키지 | cuda-compat-13-0=580.178.04-1ubuntu1 |
| 라이브러리 디렉터리 | /usr/local/cuda-13.0/compat |

NVIDIA의 [forward compatibility 문서](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)는 데이터센터 GPU 대상 지원과 R570 + cuda-compat-13-0 조합을 명시한다. 설치 전 APT simulate로 이 패키지 하나만 추가되고 기존 패키지 업그레이드·삭제가 없음을 확인했다. 컨테이너의 NVIDIA CUDA 저장소를 사용했다.

```sh
apt-get update
apt-get --simulate install --no-install-recommends cuda-compat-13-0
apt-get install --yes --no-install-recommends cuda-compat-13-0=580.178.04-1ubuntu1
```

설치된 것은 사용자 영역 라이브러리다. 호스트 커널 드라이버 설치·교체, 재부팅, GPU 대여·인스턴스 생성은 수행하지 않았다. Python 의존성 및 uv.lock은 이번 작업에서 변경하지 않았다.

## 저장소 변경

- `scripts/cuda-compat-env` 추가: R570 드라이버와 설치된 CUDA 13.0 compat libcuda.so.1을 확인하면 LD_LIBRARY_PATH 앞에 해당 경로를 추가하고 HB_CUDA_COMPAT_DIR을 지정한다. 기존 LD_LIBRARY_PATH는 보존한다.
- `scripts/preflight`, `smoke`, `extract`, `generate`, `run`, `score`, `transfer`: 공통 환경 스크립트를 source한 뒤 기존 명령 실행.
- R580 이상, 다른 드라이버 branch, 호환 라이브러리 부재 시 자동 적용하지 않는다. 확인되지 않은 조합으로 확대하지 않는다.
- `README.md`: 설치 방법·일반 실행 명령·직접 Python 실행 시 source 방법 추가. 이전의 현재 환경에서 CUDA 실행 불가 안내를 실제 검증 결과로 갱신.
- 원래 outputs 및 outputs_github의 snapshot·벡터·응답은 보존했다.

## 검증

| 항목 | 결과 |
|---|---|
| APT 설치 계획·실제 설치 | 호환 패키지 1개 추가, 기존 드라이버 변경 없음 |
| float32 CUDA 행렬곱 | A40에서 실행·synchronize 성공, CPU 결과와 정확 일치 |
| GPU 스크립트 7개 shell syntax 및 helper 적용 확인 | 통과 |
| 모의 R570 환경 | compat 경로가 앞에 추가되고 기존 library path 보존 |
| 모의 R580 환경 | 기존 library path 유지, compat 미적용 |
| `./scripts/check` | 통과: HarmBench 300, severity 300, benign safe 250, 5-fold, 외부 harmful 46·benign 99 |
| `uv run --extra gpu --extra dev ruff check .` | 통과 |
| `./scripts/preflight --model model_1` | 종료 코드 0, CUDA true, A40, bf16_supported true, Qwen revision 조회 성공 |
| helper source 후 `uv run --extra gpu --extra dev pytest` | 220 passed, 1 skipped (5.55초) |
| CUDA bfloat16 scaled_dot_product_attention | 출력 shape (1, 2, 16, 64), 유한값·synchronize 확인 통과 |

preflight의 검증 당시 free VRAM은 44.17 GiB, total VRAM은 44.43 GiB였다. Qwen ID는 Qwen/Qwen2.5-7B-Instruct, revision은 a09a35458c702b33eeacc393d103063234e8bc28, architecture는 qwen2, context window는 32768이다.

위 GPU 연산 smoke는 작은 tensor 연산 검증이다. **Qwen 전체 가중치 로딩·프로젝트의 생성 smoke·direction 재추출·전체 OOF 실험은 실행하지 않았다.** outputs_github와 activation 원소가 일치하는지는 재추출로 따로 확인해야 한다.

## 실행과 재개

현재 컨테이너에서는 다음 일반 명령으로 실행한다.

```sh
uv sync --frozen --extra gpu --extra dev
./scripts/preflight --model model_1
```

Python·pytest를 직접 실행하려면 저장소 루트에서 환경을 적용한다.

```sh
. ./scripts/cuda-compat-env
uv run --extra gpu --extra dev pytest
```

새 컨테이너에서는 NVIDIA 저장소 등록 및 위 호환 패키지 설치를 다시 확인한다. preflight 통과 후 프로젝트 smoke를 실행할 때는 기존 run snapshot과 혼합하지 않도록 새 run ID를 사용한다.

```sh
./scripts/smoke --model model_1 --run-id smoke-model1-compat13-001
```

마지막 smoke 명령은 안내용이며 이번 작업에서 실행하지 않았다. 설치한 compat 버전은 참조 outputs_github 로그에 기록되어 있지 않아, 참조와 완전히 같은 드라이버 환경이라고 주장하지 않는다. 현재 target_environment.json은 Python/PyTorch/CUDA 등을 기록하지만 compat 패키지 버전 필드는 없으므로 새 run의 환경 기록에 이번 패키지 버전을 별도로 보존해야 한다.
