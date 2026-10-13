# 팀 실험 실행 가이드

## 실험 분담

각 모델 계열을 두 명씩 맡아 L1/L2/L3 전체 평가를 함께 진행합니다. 확정된 배정은 다음과 같습니다.

| 모델 계열 | 담당자 | 모델 key | 담당 범위 |
| --- | --- | --- | --- |
| Qwen | 민정 · 승민 | `model_1` | OOF fold 1~5 + 외부 3×3 + safe over-refusal |
| Llama | 재민 · 도윤 | `model_2` | OOF fold 1~5 + 외부 3×3 + safe over-refusal |
| Gemma | 세진 · 한비 | `model_3` | OOF fold 1~5 + 외부 3×3 + safe over-refusal |
| 공동 검토 | 전체 팀원 | 모든 모델 | 레벨 근거·설정 고정·응답 표본 검수·최종 집계 |

각 조에서 실행·검수 담당을 정하고 같은 모델/run ID의 실험 명령은 한 번에 하나만 실행합니다. 같은 출력 폴더에서 fold나 벡터 레벨을 나누어 동시에 실행하면 모델 단위 잠금과 충돌합니다. 서로 다른 모델 조는 병렬 실행할 수 있고, 로그·응답의 읽기 검수는 함께 진행할 수 있습니다.

현재 확정된 것은 모델 계열과 담당자입니다. 각 계열의 버전·파라미터 크기·정확한 Hugging Face checkpoint ID와 revision은 본실험 전에 따로 확정합니다.

L3 입력에 L1 벡터를 적용하는 실험은 `source_level=1`, `target_level=3`입니다. 각 모델이 만드는 내부와 외부 3×3에는 이런 교차 조합을 모두 포함합니다. baseline에는 벡터를 적용하지 않습니다.

## 실행 전에 고정할 내용

팀 전체가 같은 Git commit, 설정 파일, 데이터 commit/hash, severity 라벨, fold, run ID를 사용합니다. 각 Pod에서 target 모델 revision은 전체 SHA로 pin하고 그 설정을 팀에 공유합니다. 모델별 tokenizer/chat template, dtype, decoder layer, coefficient, normalize, 입력·생성 길이, sampling, seed, 채점 규칙 버전을 고정합니다.

HarmBench와 외부 레벨은 프로젝트의 AI 초기 라벨입니다. 평가 전에 경계 문항의 라벨·근거를 검토하는 것이 좋습니다. 라벨 변경을 결정하면 데이터 준비를 다시 실행하고 새 run ID를 사용합니다. 외부 및 held-out 결과를 본 뒤 설정을 바꾸는 경우 해당 결과를 독립 test로 해석하지 않습니다. 설정 탐색에는 XSTest dev 등 별도의 개발 데이터를 사용합니다.

기본 채점 설정은 `method="rule"`, `version="gcg_refusal_v1"`입니다. GPU에는 target만 올리고, rule scorer는 CPU에서 실행합니다. 별도의 HarmBench 13B classifier를 다운로드할 필요가 없습니다.

## Runpod 계정과 Pod

개인 계정을 Team으로 전환한 다음 각 팀원을 자기 계정으로 초대할 수 있습니다. 각 팀원이 Pod를 직접 생성·시작·정지·삭제하려면 `Dev` 권한을 사용합니다. `Dev`는 결제 정보 조회와 결제수단 관리를 할 수 없습니다. 메뉴·권한의 현재 기준은 [Runpod 공식 계정 관리 문서](https://docs.runpod.io/accounts-billing/manage-accounts)를 확인합니다. 이 저장소는 계정 전환이나 결제 변경을 수행하지 않습니다.

팀에서 담당 모델·Pod 이름·GPU 종류·지속 볼륨·예산·종료 담당자를 정합니다. Pod 예시는 `l123-model1`, `l123-model2`, `l123-model3`입니다. 각자 자기 Pod의 target 모델 preflight와 smoke를 완료한 뒤 본실험을 실행합니다. 생성 작업이 끝나면 GPU Pod를 정지하고, 삭제 전에는 출력이 지속 저장소나 백업에 있는지 확인합니다.

모델 캐시와 `outputs/`를 지속 볼륨에 두면 Pod 재생성 후 같은 명령으로 재개할 수 있습니다. 같은 모델/run ID의 출력에 두 Pod가 동시에 쓰지 않습니다. 서로 다른 모델은 경로가 분리되므로 병렬 실행할 수 있습니다. Hugging Face token, Runpod credential, `.env`는 Git이나 결과 공유 파일에 넣지 않습니다.

## 팀원별 실행 명령

저장소를 clone한 뒤 팀에서 합의한 commit을 사용합니다. `pyproject.toml`의 `[tool.hb-steering.models]`에 세 모델 ID와 고정 revision을 공유해 두고 자신의 모델 key만 명령에서 바꿉니다. 아래는 Qwen 조(민정·승민)의 예시입니다. Llama 조는 `model_2`, Gemma 조는 `model_3`를 사용합니다.

```sh
./scripts/runpod-setup
./scripts/download
./scripts/check
./scripts/preflight --model model_1
./scripts/smoke --model model_1 --run-id smoke-model1-001
./scripts/run --model model_1 --run-id l123-rule-001
```

smoke는 fold 1의 작은 추출·생성만 확인합니다. 본실험 `run`은 아래를 순서대로 처리합니다.

1. 각 OOF fold의 fit 데이터로 L1/L2/L3 벡터를 추출하고 해당 fold held-out 문항에 baseline·세 벡터를 적용해 응답을 생성합니다.
2. HarmBench 전체 300개와 XSTest fit 150개에서 최종 벡터 세 개를 별도로 추출합니다. 고정 최종 벡터를 외부 JailbreakBench 유해 문항과 XSTest test·JailbreakBench benign에 적용해 응답을 생성합니다.
3. target 모델을 해제한 뒤 내부·외부·safe 응답 모두를 같은 CPU 규칙으로 채점합니다.
4. fold별·전체 OOF 3×3, 외부 3×3와 각 baseline, safe 데이터셋별 over-refusal을 별도로 집계합니다.

최종 벡터는 OOF fold 벡터와 구분하며 모델마다 한 번 만듭니다. 외부나 safe test 문항을 추출에 사용하지 않습니다. `Source=Original` 외부 유해 후보 55개와 benign 후보 100개를 HarmBench 전체 및 XSTest fit과 비교해 정규화한 동일 문항·유사 문구를 제외합니다. 문구 유사도는 `SequenceMatcher` 비율 0.85를 기준으로 하며, 일부 외부 유해 문항은 annotation CSV의 AI 검토 근거로 HarmBench와의 의미 중복을 명시적으로 제외합니다. 기본 최종 분모는 외부 유해 **46개(L1 9 / L2 28 / L3 9)**와 benign **99개**입니다. 라벨·screen 설정이 바뀌면 분모도 달라질 수 있습니다. 모든 의미 중복을 검출하는 절차는 아닙니다. 최종 문항 수·제외 ID·이유·레벨별 분모는 `data/processed/external/manifest.json`과 보고서에 남습니다.

긴 생성 작업과 CPU 채점을 단계별로 실행할 수도 있습니다. `--fold`를 생략한 각 단계에는 활성화된 외부·safe 단계도 포함됩니다.

```sh
./scripts/extract --model model_1 --run-id l123-rule-001
./scripts/generate --model model_1 --run-id l123-rule-001
./scripts/score --model model_1 --run-id l123-rule-001
./scripts/summarize --model model_1 --run-id l123-rule-001
```

외부·safe 평가만 따로 실행하려면 다음 명령을 사용합니다. 같은 run의 완료된 항목은 재사용합니다.

```sh
./scripts/transfer --model model_1 --run-id l123-rule-001
```

## 중단, 부분 실행, 재개

실행이 중단되거나 일부 생성/채점이 실패하면 **같은 설정과 같은 명령**으로 재실행합니다. 완료된 응답·판정은 검증 후 재사용하고 실패 항목만 다시 수행합니다. 실패를 ASR_rule 0이나 정상 거부로 집계하지 않습니다.

다음은 내부 한 fold만 제한해서 실행하는 예시입니다.

```sh
./scripts/run --model model_1 --run-id partial-model1-001 --fold 1 --limit-per-level 2
```

`--fold N`은 내부 OOF만 수행하고 최종 외부 단계는 수행하지 않습니다. `--fold` 없이 한도를 지정하면 OOF와 외부·safe를 함께 제한합니다. 한도 `N`은 유해 fit/eval에 레벨당 `N`, benign fit과 각 safe 데이터셋 평가에 최대 `3N`을 사용합니다. 선택은 ID 정렬로 결정적입니다. 제한된 결과는 `full_oof=false`/`full_transfer=false`로 표시하며 본실험과 run ID를 분리합니다. 설정·한도·라벨·코드가 달라졌다면 새 run ID로 시작합니다.

## 결과 해석과 팀 전달물

```text
ASR_rule = 규칙상 비거부 응답 수 / 기대 평가 수
safe over_refusal = 1 - ASR_rule
                  = 규칙상 거부 응답 수 / 기대 평가 수
```

유해 입력의 ASR_rule은 실제 유해 내용 수행의 성공률이 아닙니다. 거부 표현이 없어도 무관한 답변일 수 있고, 정상적인 설명 중의 거부 표현도 오탐될 수 있습니다. 규칙 버전·hash를 통일하고 원문 응답 표본을 함께 검수합니다. safe 입력의 ASR_rule은 비거부율이며 over-refusal이 그 보완값입니다.

| 보고서 | 표 구성 | 적용 벡터 |
| --- | --- | --- |
| HarmBench OOF | 3×3 + L1/L2/L3 baseline | 해당 fold의 fit에서 만든 세 벡터 |
| 외부 유해 | 3×3 + L1/L2/L3 baseline | HarmBench 전체에서 만든 고정 세 벡터 |
| XSTest safe test | baseline + L1/L2/L3 over-refusal | 같은 고정 세 벡터 |
| JailbreakBench benign | baseline + L1/L2/L3 over-refusal | 같은 고정 세 벡터 |

safe 문항에 위해도 레벨은 붙이지 않습니다. baseline과 같은 입력·생성 설정을 사용하고 두 safe 데이터셋은 각각 보고합니다. 여러 layer를 설정했다면 layer마다 별도 표가 나오므로 같은 layer 조건끼리 비교합니다.

각 팀원은 다음을 지속 볼륨이나 팀의 합의된 파일 저장소에 보존합니다. 이 저장소는 파일을 팀원에게 자동 전송하지 않습니다.

- `outputs/<run_id>/`의 설정 snapshot·manifest·code hash
- 담당 모델의 모든 `fold_<N>/` 벡터·원문 응답·개별 판정·로그
- `<model_key>/metrics/`의 OOF `summary.json`, `asr_rule.csv`, `asr_rule.md`와 기존 호환 파일
- `<model_key>/transfer/`의 최종 벡터·원문 응답·개별 판정·외부 `asr_rule.*`·`over_refusal.*`·`summary.json`
- 실행한 Git commit, Pod/GPU 정보, preflight/smoke 결과, 실패·재시도 및 사람 검수 기록

완료 판단은 프로세스가 종료됐는지뿐 아니라 `complete`, `full_oof`/`full_transfer`, 기대 수·생성 수·판정 수가 맞는지 확인해서 합니다. 미완료 비율은 `null`이며 작은 partial 결과를 전체 결과로 합치지 않습니다. OOF는 fold별 비율의 단순 평균 대신 비거부 수 합계/분모 합계로 비교합니다. 외부 결과는 고정 최종 벡터의 평가로 OOF와 구분합니다.
