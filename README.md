# HarmBench level steering

HarmBench 문항을 프로젝트의 L1/L2/L3로 나누고, 각 레벨의 activation 평균에서 benign 평균을 빼서 steering vector를 만듭니다. **그룹 기반 5-fold OOF 평가**로 모든 문항을 한 번씩 held-out 평가에 사용하며, 모델마다 **벡터를 만든 레벨 × 평가하는 레벨의 3×3 ASR 표**와 steering 전 baseline을 저장합니다.

목표는 읽기 쉬운 코드, 명시적인 설정, 작은 실행을 통한 검증, 실패 원인 기록, 중단 후 재개입니다. 모델 3개의 Hugging Face ID는 나중에 `configs/experiment.toml`에 채우면 됩니다. GPU를 대여하거나 Runpod 인스턴스를 만드는 기능은 포함하지 않습니다.

## 바로 시작

Python은 `uv`가 프로젝트의 3.11 버전을 관리합니다. `uv`가 PATH에 있어야 합니다.

```sh
cd ~/Desktop/harmbench-steering
./scripts/download
./scripts/check
```

`./scripts/download`는 `#!/usr/bin/env -S uv run -s`와 PEP 723 metadata를 가진 실행 파일입니다. 별도 GPU 의존성 없이 다음을 한 번에 처리합니다.

1. 커밋과 SHA-256이 고정된 HarmBench 및 XSTest 원본 다운로드
2. 번들된 `data/annotations/severity.csv`의 레벨과 근거 적용
3. seed 42로 그룹 기반 5-fold 생성 및 검증

이미 검증된 다운로드와 fold가 있으면 재사용합니다. 원본, 라벨, 분할 설정의 변경을 감지하면 조용히 덮어쓰지 않습니다. 새 clone은 Git에 포함된 고정 fold를 그대로 사용합니다. 기존 fit/dev/test 데이터가 있는 작업 폴더는 업데이트 후 한 번만 아래 명령으로 재준비합니다.

```sh
./scripts/download --resplit
```

라벨이나 분할 설정을 의도적으로 바꿀 때도 같은 명령을 사용하고 변경 내용을 Git에서 확인합니다. 기존 실행 결과에는 새 fold를 섞지 말고 새 run ID를 사용합니다.

## Runpod에서 실행

프로젝트를 GPU Pod로 복사하거나 Git으로 clone한 뒤 프로젝트 디렉터리에서 실행합니다. GPU 드라이버가 있는 Linux 환경과 충분한 VRAM이 필요합니다. 필요한 VRAM은 나중에 고를 target 모델과 13B judge에 따라 달라집니다.

현재 `uv.lock`의 Linux PyTorch 패키지는 CUDA 13 계열입니다. 일반적인 실행 환경은 NVIDIA R580 이상 드라이버가 필요하므로 Pod의 드라이버를 확인합니다. `preflight`가 CUDA 사용 가능 여부를 확인합니다. [NVIDIA 호환성 표](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

```sh
./scripts/runpod-setup
./scripts/download
./scripts/check
```

`runpod-setup`은 `uv`가 없을 때만 `python3 -m pip install uv`로 설치하고, `uv sync --frozen --extra gpu --extra dev`와 환경 preflight를 실행합니다. 모델을 아직 정하지 않았어도 환경 점검까지 할 수 있습니다.

`configs/experiment.toml`에서 다음 세 ID를 채웁니다.

```toml
[models.model_1]
hf_id = "organization/model-name"
revision = ""
dtype = "auto"
```

`model_2`, `model_3`도 같은 방식으로 입력합니다. 필요한 경우 Hugging Face 접근 권한을 받은 뒤 셸에 `HF_TOKEN`을 설정합니다. `.env`는 자동 로딩하지 않습니다. 파일을 쓰려면 직접 `uv run --env-file .env ...`로 실행합니다.

기본 입력 형식은 tokenizer의 chat template입니다. base 모델처럼 chat template이 없는 모델은 해당 모델 설정에 `use_chat_template = false`를 명시합니다. 실제 decoder 구조 지원 여부는 preflight와 smoke에서 확인합니다.

```sh
./scripts/pin-models
./scripts/preflight --model model_1
./scripts/smoke --model model_1 --run-id smoke-001
```

`pin-models`는 입력된 target ID와 공식 judge ID의 revision을 전체 commit SHA로 고정합니다. 빈 target ID는 건너뜁니다. smoke는 기본적으로 fold 1에서 fit과 held-out 평가를 레벨당 최대 2개 문항으로 제한해 벡터 추출과 생성을 확인하며 생성 길이는 최대 16개 token입니다. judge는 실행하지 않습니다. 이 단계를 먼저 확인하고 전체 실행으로 넘어갑니다.

```sh
./scripts/run --model model_1 --run-id oof-001
./scripts/run --model model_2 --run-id oof-001
./scripts/run --model model_3 --run-id oof-001
```

각 명령은 기본적으로 fold 1~5를 순서대로 처리합니다. 같은 run ID 아래에서도 모델과 fold별 출력 디렉터리를 분리합니다. 각 명령은 프로젝트 경로를 자체적으로 찾으므로 다른 디렉터리에서 절대 경로로 호출해도 됩니다.

## 5-fold 실험 흐름

각 fold에서 다른 네 fold를 `fit`, 남은 한 fold를 `test`로 사용합니다. 그 fold의 벡터는 fit 문항으로만 계산하고 held-out test에서 평가합니다. 이 과정을 다섯 번 반복하면 모든 HarmBench 문항이 정확히 한 번씩 평가됩니다. 같은 그룹의 문항이 fit과 test에 나뉘지 않도록 검증합니다.

레이어·계수·정규화 등은 전체 OOF 실행 전에 고정합니다. held-out 결과를 보고 설정을 바꾸면 그 결과를 독립 평가로 해석할 수 없습니다. 설정 탐색이 필요하면 별도 dev 데이터나 nested CV를 사용해야 합니다. 현재 명령은 고정 설정의 OOF 평가이며 별도 dev split을 만들지 않습니다. 기본값이 최적이라는 주장은 하지 않습니다.

```sh
# 한 fold의 작은 실행으로 파이프라인 확인
./scripts/run --model model_1 --run-id partial-001 --fold 1 --limit-per-level 2

# 고정 설정으로 전체 OOF 실행
./scripts/run --model model_1 --run-id oof-002
```

`--fold N`은 한 fold만 처리하는 옵션입니다. 이를 생략하면 전체 5개 fold를 처리합니다. `--limit-per-level`은 fold별 fit과 test 문항 수를 제한하는 부분 실행 옵션입니다. 제한이 있는 결과는 `full_oof=false`로 표시하고 전체 OOF 분모를 갖춘 결과로 보고하지 않습니다. 부분 실행과 전체 실험은 run ID를 분리합니다. 같은 ID의 한도·설정·데이터를 바꾸면 기존 실행과 불일치 오류가 납니다. `--split` 옵션은 사용하지 않습니다.

필요하면 단계를 나눠 실행합니다. 전체 fold의 target 작업을 마치고 모델 메모리를 해제한 뒤 judge를 로딩하므로 target과 judge를 동시에 올릴 필요는 없습니다.

```sh
./scripts/extract --model model_1 --run-id oof-002
./scripts/generate --model model_1 --run-id oof-002
./scripts/score --model model_1 --run-id oof-002
./scripts/summarize --model model_1 --run-id oof-002
```

각 단계에도 `--fold N`을 지정할 수 있습니다. `summarize --fold N`은 해당 fold 보고서를 작성하면서 현재 저장된 전체 fold의 모델별 집계도 갱신합니다. 모든 필수 fold와 평가 항목이 완료되기 전에는 집계가 `complete=false`로 표시됩니다. `run`과 `summarize`의 종료 상태는 `--fold N`이 있으면 선택한 fold의 완료 여부를, 생략하면 전체 OOF 완료 여부를 따릅니다. 따라서 한 fold 명령이 성공해도 모델 전체 보고서는 아직 미완료일 수 있습니다.

사용 가능한 인자는 각 실행 파일의 `--help`로 확인합니다. 별도 설정 파일을 쓰려면 전역 `--config`를 subcommand 앞에 둡니다.

```sh
uv run --extra gpu hb-steering --root "$PWD" --config configs/experiment.toml \
  run --model model_1 --run-id custom-001
```

## 데이터와 레벨

- **HarmBench:** 고정 원본의 400개 text behaviors 중 standard 200개와 contextual 100개를 사용합니다. copyright 100개와 multimodal 데이터는 기본 실험에서 제외합니다. contextual 문항은 context를 입력 및 판정에 포함합니다.
- **레벨:** L1은 제한적인 피해, L2는 상당한 피해, L3는 생명·중대한 신체 피해 또는 대규모 피해를 중심으로 정한 프로젝트 분류입니다. `data/annotations/severity.csv`는 ID·레벨·근거를 담은 **AI가 작성한 초기 라벨**이며 HarmBench의 공식 레벨이나 사람이 검수한 확정 라벨이 아닙니다. 이 분류로 얻는 결과는 해당 라벨 기준의 비교입니다. 본실험 전에 경계 문항과 근거를 검토하는 것이 좋습니다.
- **Benign 기준:** XSTest의 safe 250개 중 고정된 fit 150개를 모든 fold의 benign 평균 계산에 사용합니다. 나머지 dev 50개/test 50개는 보존하지만 현재 HarmBench ASR 계산에는 사용하지 않습니다. XSTest는 과도한 거부를 점검하도록 설계된 safe 문항으로, HarmBench와 내용이 완전히 대응하는 중립 대조군은 아닙니다.
- **분할:** 기본값은 `n_folds = 5`, seed 42입니다. HarmBench는 레벨별 분포를 고려해 그룹을 통째로 배치합니다. Unicode·대소문자·공백을 정규화한 동일 behavior 또는 동일한 비어 있지 않은 context로 연결되는 문항은 항상 같은 fold에 둡니다. 그룹 크기에 따라 fold별 문항 수가 달라질 수 있습니다. 의미가 비슷한 다른 표현까지 자동으로 찾아내지는 않습니다.

번들된 초기 라벨의 전체 OOF 평가 문항 수는 다음과 같습니다. 모든 fold를 완료했을 때 각 문항이 한 번씩 평가되므로 3×3 표의 각 열은 해당 레벨의 전체 문항 수를 분모로 갖습니다. 레벨별 문항 수가 같지 않으므로 ASR을 읽을 때 분모도 함께 확인합니다.

| 평가 레벨 | 전체 문항 | 전체 OOF 평가 분모 |
| --- | ---: | ---: |
| L1 | 29 | 29 |
| L2 | 189 | 189 |
| L3 | 82 | 82 |
| 합계 | 300 | 300 |

기본 fold별 문항 수는 다음과 같습니다. 각 칸은 **fit / test**입니다.

| fold | L1 | L2 | L3 |
| --- | ---: | ---: | ---: |
| 1 | 23 / 6 | 151 / 38 | 65 / 17 |
| 2 | 23 / 6 | 151 / 38 | 65 / 17 |
| 3 | 23 / 6 | 151 / 38 | 66 / 16 |
| 4 | 23 / 6 | 151 / 38 | 66 / 16 |
| 5 | 24 / 5 | 152 / 37 | 66 / 16 |

300개 문항은 285개 그룹으로 묶입니다. fold별 fit·test 수와 전체 그룹 수는 `data/processed/summary.json`에도 저장됩니다. L1은 전체를 평가해도 29개이므로 레벨 간 작은 차이를 과도하게 해석하지 않습니다.

데이터 출처: [HarmBench](https://github.com/centerforaisafety/HarmBench), [XSTest](https://github.com/paul-rottger/xstest). 버전·파일 hash는 설정과 다운로드 manifest에 남깁니다. 라이선스 및 인용 정보는 `third_party/NOTICE.md`에 있습니다.

## 벡터와 ASR

각 target 모델의 각 fold와 각 레벨에 대해 다음을 계산합니다.

```text
v[fold, level] = mean(해당 fold의 level별 fit activation)
                 - mean(해당 fold의 benign fit activation)
```

기본 추출 위치는 **마지막 decoder block의 output, 마지막 prompt token**입니다. 기본 개입은 같은 block output에 `coefficient × vector`를 더하며 prompt와 생성 token 모두에 적용합니다. `layers = [-1]`, `coefficient = 1.0`, `normalize = false`가 시작 설정입니다. 모델 간 hidden 크기가 달라도 각 모델에서 자체 벡터를 계산합니다. 모델 사이로 벡터를 옮기는 실험은 아닙니다.

| 벡터 계산 레벨 ↓ / 평가 레벨 → | L1 | L2 | L3 |
| --- | --- | --- | --- |
| L1 | ASR | ASR | ASR |
| L2 | ASR | ASR | ASR |
| L3 | ASR | ASR | ASR |

fold마다 이 표 하나와 레벨별 baseline 3개를 저장하고, **모든 fold의 성공 수 합계 / 기대 평가 수 합계**로 모델별 OOF 표를 만듭니다. 필요한 생성과 판정이 모두 완료되어 판정 수가 기대 평가 수와 일치할 때만 ASR을 제공합니다. fold별 ASR을 단순 평균하지 않습니다. 모든 모델과 fold가 완료되면 모델별 최종 표 3개, steering 조합 27개, baseline 9개가 나옵니다. 각 held-out 문항은 baseline과 같은 fold에서 만든 세 벡터 조건에 동일하게 사용합니다.

이 결과는 **fold별 벡터를 만드는 절차의 전이 성능**입니다. fold마다 벡터가 다르므로 고정된 최종 벡터 하나의 성능으로 해석하지 않습니다. 모든 데이터로 최종 벡터를 다시 만들 경우 그 벡터의 독립 성능은 별도 평가 데이터로 확인해야 합니다.

`layers`에 여러 위치를 지정하면 각 레이어에 독립적으로 개입하고 **레이어별 3×3 표**를 만듭니다. 여러 레이어를 동시에 개입하는 설정은 아닙니다.

ASR은 공식 `cais/HarmBench-Llama-2-13b-cls`와 standard/contextual 판정 프롬프트를 사용합니다. 생성 오류, judge 오류, 예상하지 못한 judge 출력은 정상적인 `no`로 처리하지 않습니다. 결과를 읽을 때 성공 수(`n_success`), 생성 수(`n_generated`), 판정 수(`n_scored`), 기대 평가 수(`n_expected`), 그룹 수(`n_groups`)와 완료 여부(`complete`)를 확인합니다. 누락이 있는 결과의 ASR과 구간은 `null`이며 오류 원인은 logs에서 확인합니다. 빠진 판정이 있으면 전체 OOF 결과가 완료된 것으로 표시하지 않습니다.

보고서의 Wilson 95% 구간은 문항이 서로 독립이라는 이항 가정 아래 계산한 참고 구간입니다. 같은 context를 공유하는 문항의 상관이나 fold 간 fit 데이터 중첩에 따른 불확실성을 모두 반영하지 않습니다. 공식 구성 요소를 재사용하지만 프로젝트의 레벨·fold·steering 설정을 추가한 실험이므로 전체 공식 HarmBench leaderboard 결과와 같은 조건의 점수는 아닙니다.

## 출력과 재개

```text
outputs/<run_id>/
├── config.toml                # 원본 TOML 설정 snapshot
├── effective_config.json      # smoke 제한 등을 반영한 실제 설정
├── manifest.json              # fold·한도·설정·데이터·코드 식별 정보
├── code_hashes.json            # 소스 파일별 SHA-256
└── <model_key>/
    ├── fold_1/                # fold_2~fold_5도 같은 구조
    │   ├── vectors/           # 이 fold의 fit으로 만든 레벨별 벡터
    │   ├── generations/       # held-out baseline·steering 응답
    │   ├── metrics/           # judge 결과·fold별 ASR
    │   └── logs/              # 오류 및 실행 정보
    └── metrics/
        ├── summary.json       # 전체 fold 완료 여부·OOF 집계
        ├── asr.csv            # 성공 수·분모·구간·그룹 수
        └── asr.md             # 모델별 OOF 3×3 표·baseline
```

실행 설정, fold 파일 hash, 데이터 정보와 소스 코드 hash는 run에 고정됩니다. 원본 TOML과 실제 적용 설정을 따로 남기므로 smoke 제한도 확인할 수 있습니다. 중단 후 **같은 명령**을 다시 실행하면 완료된 작업을 재사용합니다. 일부 fold만 완료된 상태에서 같은 run의 남은 fold를 이어 실행할 수 있습니다. 실패한 항목의 재시도 기록은 결과와 구분해서 보존합니다. 마지막 journal 줄이 프로세스 종료로 잘렸을 때만 복구하며, 중간 데이터 손상은 명시적으로 실패합니다. 같은 모델과 run ID에 동시에 쓰는 실행은 파일 잠금으로 차단합니다. 코드를 수정한 뒤에는 새 run ID를 사용합니다.

`outputs/`, `data/raw/`, `data/processed/`, 모델 캐시, `.venv/`, `.env`, `work/`는 Git에서 제외합니다. 라벨, 고정 분할, 설정, 소스, 테스트, `uv.lock`은 추적합니다. Pod가 삭제돼도 필요한 결과를 잃지 않도록 `outputs/`를 지속 볼륨에 두거나 직접 백업합니다.

## 프로젝트 지도

```text
harmbench-steering/
├── AGENTS.md                  # 코드 작성·검증·실행 원칙
├── README.md                  # 구조와 실행 가이드
├── pyproject.toml             # uv 프로젝트·선택적 GPU 의존성
├── uv.lock                    # 고정 의존성
├── configs/experiment.toml    # 데이터·모델·steering·judge 설정
├── scripts/                   # download, check, smoke, run 등 진입점
├── src/hb_steering/
│   ├── cli.py                 # 명령 연결·실행 흐름
│   ├── experiment.py          # 실행 계획·벡터·생성·판정·집계 연결
│   ├── data.py                # HarmBench·라벨·그룹 기반 fold
│   ├── prepare.py             # XSTest·다운로드·고정 fold 관리
│   ├── config.py              # 설정 검증
│   ├── models.py              # HF 로딩·prompt·token 처리
│   ├── vectors.py             # activation 평균 차이
│   ├── steering.py            # hook 설치·제거
│   ├── evaluate.py            # HarmBench judge
│   ├── metrics.py             # 성공 수 집계·Wilson 구간
│   └── storage.py             # 원자적 저장·journal·잠금
├── data/
│   ├── annotations/           # 추적하는 초기 레벨·근거
│   ├── splits/                # 추적하는 고정 fold ID·metadata
│   ├── raw/                   # 다운로드 원본·manifest, Git 제외
│   └── processed/             # 파생 데이터, Git 제외
├── tests/                     # 데이터·벡터·hook·재개·판정 검증
├── third_party/               # 출처·라이선스 원문
├── outputs/                   # 실행별 산출물, Git 제외
└── work/                      # 임시 파일, Git 제외
```

## 코드 검증

```sh
uv sync --frozen --extra gpu --extra dev
uv run --extra gpu --extra dev pytest
uv run --extra dev ruff check .
```

테스트는 작은 fixture와 CPU의 작은 모델로 데이터 분리·activation 위치·개입 제거·저장·재개를 확인합니다. 실제 선택할 모델 및 Runpod GPU의 호환성·VRAM·전체 실험 성공은 각 모델의 preflight와 smoke로 별도 검증해야 합니다.
