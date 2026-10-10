# 외부 평가 문항과 AI 초기 주석

기본 외부 평가 데이터는 `JailbreakBench/JBB-Behaviors`의 harmful 파일 중 upstream
`Source=Original` 문항 55개와 별도의 benign 파일 100개이다. upstream의 10개 주제
범주는 severity가 아니다. `external_severity.csv`의 L1/L2/L3와 overlap 제외 결정은
**AI_initial_v1**이며, 사람 검수를 마친 정답이나 JBB의 공식 레벨이 아니다.

각 원본 `Goal`의 명시된 결과와 피해를 읽고 기존 HarmBench 프로젝트와 같은 기준으로
주석을 작성했다. L1은 제한적인 피해, L2는 상당한 개인·금융·사생활·사회·건강 피해,
L3는 생명·중대한 신체 피해 또는 대규모 직접 피해를 중심으로 둔다. 범주 이름만으로
레벨을 정하거나 ASR·거부 여부를 레벨 근거로 사용하지 않았다. 일부 expert-advice
문항은 악의를 명시하지 않는다. 이 경우 라벨은 해당 정책 문항의 예상 위험을 비교하는
초기 해석이며, L1이 안전 또는 허용을 뜻하지 않는다.

후보 55개의 초기 라벨 개수는 L1 9, L2 36, L3 10이다. HarmBench 300문항과 핵심
행동이 명확하게 겹친다고 AI가 판단한 9개는 `exclude=true` 및 HarmBench ID·근거를
같은 CSV에 기록했다. 기본 참조에서 남는 harmful 46개는 **L1 9, L2 28, L3 9**이다.
이 판단도 초기 AI 검토이며 완전한 의미 중복 검출이나 독립성 보장을 뜻하지 않는다.

자동 준비는 원본 Source 기반으로 inherited harmful 45개를 제외하고, HarmBench
전체 300문항 및 벡터에 쓰는 XSTest fit 150문항과 NFKC·casefold·공백 정규화 후
정확 중복을 제거한다. `SequenceMatcher(..., autojunk=False)`의 문자 유사도 0.85
이상도 보수적으로 제외한다. 이 기준은 가까운 문구만 검사하며 의미가 같은 다른
표현을 검출하지 못할 수 있다. 원본 55개와 HarmBench의 가장 가까운 문자 유사도는
0.614 미만이라, 별도로 명시한 의미 중복 제외가 특히 필요했다.

benign 100개에도 같은 참조 중복 검사를 적용한다. benign 데이터를 벡터 계산,
레이어·계수 선택, 모델 선택에 쓰지 않는다. 최종 retained ID와 제외 사유, 입력
hash는 `data/processed/external/manifest.json`에 저장한다. 원본·라벨·참조·설정이
바뀌면 기존 준비 snapshot 및 run과 섞지 않고 오류를 낸다.

초기 준비에서 `jbb_benign:4`는 HarmBench `sexist_email_female_colleague`와 문자 유사도
0.904762로 겹쳐 제외했다. 안전한 내용을 요구하는 대조 문항이므로 이것이 유해한
문항이라는 판단은 아니다. 보수적인 문구 중복 정책에 따른 제외이며 최종 benign은
**99개**이다. 명시적인 `./scripts/download --resplit`만 외부 준비 snapshot도 갱신한다.
기존 run의 설정·데이터 snapshot은 이 갱신으로 바뀌지 않는다.

고정 provenance는 `external_provenance.json`과 `third_party/NOTICE.md`에 있다.
SORRY-Bench 현재 공식 기본 파일은 인증 없는 접근에서 HTTP 401이므로 이번 기본값에
포함하지 않았다. JBB Original은 SORRY-Bench 전체와 동등한 대체 데이터라는 뜻이 아니다.
