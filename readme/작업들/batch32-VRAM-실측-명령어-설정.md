# Batch 32 구현·VRAM 실측·명령어 설정

- 날짜: 2026-10-13 (UTC)
- session id: `01a13446-d303-7f72-ab96-9719cc0287d3`
- [작업 목록](../overview.md)

## 요청과 결과

batch 증가에 따른 실제 VRAM을 측정하고, 후속 요청에 따라 실제 실험을 batch 32로 구현·설정했다. 이후 명령어로 batch size를 지정할 수 있도록 scripts/run과 CLI를 수정했다.

```sh
./scripts/run --model model_1 --run-id full-model1-batch32-20261013-001 --batch-size 32
```

이 명령은 안내용이며 전체 실험을 이번 작업에서 실행하지 않았다. 실제 실행한 범위는 아래 벤치마크 및 작은 GPU smoke다.

## 측정 조건

- GPU: NVIDIA A40, 실제 total VRAM 44.43 GiB. 사용자가 제시한 50 GiB는 여유 메모리 계산용 가정이다.
- 모델: Qwen/Qwen2.5-7B-Instruct, revision a09a35458c702b33eeacc393d103063234e8bc28.
- PyTorch 2.14.1+cu130, Transformers 4.57.6, bfloat16, SDPA, CUDA 13 forward compatibility 적용.
- 모델을 올린 뒤 allocated: 약 14.227 GiB.
- 안전한 합성 문장을 chat template으로 인코딩. 입력 512 또는 2048 token을 실제 tokenizer 길이로 확인. 같은 길이의 같은 입력을 batch 내 반복했으며 padding 비용은 없는 조건이다.
- 추출: 마지막 decoder block의 마지막 nonpadding prompt token activation을 CPU float32로 수집. use_cache=false. 모델 full forward를 조건별 3회 실행한 peak.
- 생성: baseline, use_cache=true, do_sample=false. 조기 EOS 차이를 막기 위해 min_new_tokens=max_new_tokens=512를 강제. 조건별 1회 생성.
- 각 조건 전 CUDA synchronize, cache 정리, peak 카운터 reset. 종료 후 synchronize하고 memory_allocated·memory_reserved peak 확인.
- OOM 발생 시 오류를 저장하고 중단하며 자동으로 batch·token 길이를 줄이지 않는다. 이번 최종 측정 조건은 모두 성공했다.

## 실제 측정값

단위는 GiB. allocated는 실제 PyTorch tensor 사용량, reserved는 allocator가 확보한 메모리다. 드라이버·CUDA context 및 다른 프로세스 메모리를 모두 포함하는 값은 아니다.

| 작업 | 입력 token | 출력 token | Batch | Peak allocated | Peak reserved | 모델 포함 baseline 대비 추가 allocated | 소요 시간(초) |
|---|---|---|---|---|---|---|---|
| extract_forward | 512 | 0 | 1 | 14.384 | 14.580 | 0.148 | 0.46 |
| extract_forward | 512 | 0 | 2 | 14.533 | 14.809 | 0.298 | 0.51 |
| extract_forward | 512 | 0 | 4 | 14.829 | 15.262 | 0.594 | 0.97 |
| generate | 512 | 512 | 1 | 14.331 | 14.496 | 0.095 | 15.79 |
| generate | 512 | 512 | 2 | 14.429 | 14.570 | 0.194 | 16.26 |
| generate | 512 | 512 | 4 | 14.623 | 14.756 | 0.388 | 16.57 |
| extract_forward | 2048 | 0 | 1 | 14.829 | 15.264 | 0.594 | 0.99 |
| extract_forward | 2048 | 0 | 2 | 15.423 | 16.115 | 1.188 | 1.98 |
| extract_forward | 2048 | 0 | 4 | 16.610 | 17.928 | 2.375 | 3.83 |
| extract_forward | 2048 | 0 | 32 | 33.236 | 42.922 | 19.001 | 30.12 |
| generate | 2048 | 512 | 1 | 14.623 | 14.756 | 0.388 | 16.05 |
| generate | 2048 | 512 | 2 | 14.999 | 15.297 | 0.764 | 17.09 |
| generate | 2048 | 512 | 4 | 15.764 | 16.256 | 1.529 | 18.47 |
| generate | 2048 | 512 | 32 | 26.456 | 27.688 | 12.221 | 38.49 |

추출 시간은 forward 3회 합계이고 생성 시간은 1회다. 시간 비교용 반복·통계 실험은 수행하지 않았다.

## 50 GiB에서 batch 32의 여유

2048-token 추출에서 실측한 batch 32 allocated는 33.236 GiB, reserved는 42.922 GiB다. 50 GiB 기준 각각 16.764 GiB와 7.078 GiB가 남는다. 따라서 reserved까지 기준으로 보면 처음 제시한 10 GiB 여유에는 못 미치지만, 사용자 요청대로 batch 32를 유지했다.

2048-token 입력 + 512-token 생성에서 batch 32 allocated는 26.456 GiB, reserved는 27.688 GiB다. 실제 A40에서도 두 조건 모두 OOM 없이 완료됐다. 4096-token 입력 및 길이가 섞인 batch 32는 측정하지 않았으므로 전체 데이터에서 같은 사용량이나 성공을 보장하지 않는다.

추출은 prompt 전체의 logits를 포함한 full forward를 측정하고, 생성은 라이브러리의 generation 경로를 사용하므로 추출 peak가 생성 peak보다 클 수 있다. 다른 구현·길이·padding·steering 조건으로 이 수치를 그대로 외삽하지 않는다.

## 실제 batching 구현

- configs/experiment.toml: runtime.batch_size=32.
- models.py: 설정한 크기로 activation 추출을 나누고 prompt별 activation을 원래 순서로 합산. left padding, attention mask, padding을 제외한 position_ids 사용. generate_batch에서 prompt 전체 padded 폭을 제외하고 응답만 각각 decode.
- experiment.py: baseline 또는 같은 source level·layer 조건끼리 작업을 묶어 최대 32개씩 생성. 결과 개수·비어 있지 않은 응답 검증 후 ID별 journal 저장. 완료된 항목만 재사용. batch 오류는 해당 작업별 로그에 기록. OOM은 중단.
- 마지막 batch가 32보다 작은 것은 남은 문항 수 때문이며 OOM 회피용 자동 축소가 아니다.
- batch size에 따라 연산·padding이 달라지므로 batch 1의 벡터와 원소 단위 일치를 보장하지 않는다. 서로 다른 batch 설정은 새 run ID로 구분한다.

## 명령어 설정

scripts/run의 최종 실행 줄은 다음과 같다.

```sh
exec uv run --no-sync --project "$ROOT" --extra gpu hb-steering --root "$ROOT" run --batch-size 32 "$@"
```

기본 batch는 32이고 사용자 인자가 뒤에 붙는다. 원하는 값으로 바꾸려면 명령 끝에 --batch-size를 지정한다. CLI override는 setup_run 전에 유효 config에 반영되어 snapshot·fingerprint에 포함된다. 같은 run ID에서 batch 설정이 바뀌면 재개를 거부한다. extract·generate·score·summarize·transfer·smoke도 --batch-size를 지원한다.

```sh
./scripts/run --model model_1 --run-id full-model1-batch32-20261013-001 --batch-size 32
```

위 전체 실행 명령은 --fold·--limit-per-level을 생략해 모든 내부 fold와 활성화된 외부·safe 단계를 처리한다. 기존 smoke run ID를 전체 실험에 재사용하지 않는다.

## 검증

- ./scripts/check 통과, 모델 preflight 통과.
- 전체 pytest: 227 passed, 1 skipped (6.22초). padding이 섞인 batch activation 평균, 짧은 마지막 batch, 응답 token 분리·순서, hook 예외 제거, chunk 오류·재개 테스트 포함.
- 이후 batch size의 0·음수·boolean·소수 설정 거부를 추가하고 해당 설정 테스트 20개 통과.
- 이후 CLI --batch-size 설정 전달·잘못된 값 거부·wrapper 기본값 덮어쓰기 테스트 추가. tests/test_cli.py 19개 통과.
- lint 및 git diff 공백 검사 통과.
- GPU smoke: smoke-model1-batch32-20261013-001에서 fold 1 벡터 세 개 추출·24개 응답 생성 완료. smoke 표본은 레벨당 2개, benign fit 6개이므로 실제 batch는 최대 6이었다. 실제 32개 입력은 별도 벤치마크에서 확인했다. judge·ASR 평가는 실행하지 않았다.
- CLI 옵션 추가 이후 전체 GPU smoke를 반복하지 않았다. 옵션 전달은 CLI 테스트로 검증했고 배치 모델 코드는 변경되지 않았다.

## 원본 결과와 재현

- [원본 VRAM 결과 vram-model1-20261013-002](../../outputs/vram-model1-20261013-002/model_1/diagnostics/vram.json)
- [원본 VRAM 결과 vram-model1-batch32-20261013-001](../../outputs/vram-model1-batch32-20261013-001/model_1/diagnostics/vram.json)

초기 벤치마크 vram-model1-20261013-001은 입력 반복 수의 초기 추정 때문에 실제 길이가 414·1438 token이었다. 해당 원본을 보존하고 tokenizer 길이를 이진 탐색하는 방식으로 보정한 뒤 새 run ID에서 512·2048 token 측정을 수행했다. 위 최종 표는 보정 후 결과와 batch 32 결과만 사용한다.

```sh
./scripts/benchmark-vram --model model_1 --run-id vram-model1-repeat-001 --batch-sizes 1 2 4 --input-targets 512 2048
./scripts/benchmark-vram --model model_1 --run-id vram-model1-batch32-repeat-001 --batch-sizes 32 --input-targets 2048
```

다른 설정으로 실행할 때는 새 run ID를 사용한다. 결과는 outputs/<run_id>/<model_key>/diagnostics/vram.json에 원자 저장하며 같은 경로 동시 실행은 lock으로 막는다. 기존 OOF·참조 산출물은 변경하지 않았다.
