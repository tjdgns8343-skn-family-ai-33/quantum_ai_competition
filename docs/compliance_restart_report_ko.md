# QML 대회 규정 준수 재시작 보고서

작성 기준일: 2026-08-04  
대상 저장소: `quantum_ai_competition`

## 1. 재시작 결론

이 저장소는 과거 B22~B46 계열의 코드·weights·artifact·보고서를 복사하지 않고 빈 저장소에서 시작했다. 과거 최고 모델 B46은 raw feature를 QASM 내부에서 받았지만, Logistic Regression으로 만든 B30 계수와 label 기반 ridge Cauchy codebook amplitude를 `theta_N`으로 이전했다. 추가 공지는 최종 QASM 형식과 무관하게 이 parameter-generation 과정을 금지하므로 B46은 대회 제출 후보에서 제외한다.

과거 결과는 연구 기록으로만 기존 저장소에 남고, 이 저장소의 학습 및 제출 과정에는 사용하지 않는다.

## 2. 적용한 여섯 가지 원칙

1. B22~B46 및 그 파생 weights를 제출 후보에서 완전히 제외한다.
2. B1 direct-raw VQC를 label과 무관한 무작위 초기값에서 새로 학습한다.
3. 초기값은 지정한 seed의 uniform random 값이며 외부 weight 파일이나 warm start 입력을 받지 않는다.
4. 모든 `theta_0~theta_15`는 `EstimatorQNN`이 같은 양자 회로에서 계산한 정확한 q0 확률과 train label 사이의 balanced binary cross-entropy만으로 COBYLA가 갱신한다. validation은 최대 허용량인 1,024-shot SamplerQNN으로 평가한다.
5. Logistic Regression, SVM, QDA, boosting, classical neural-network teacher, ridge codebook 등은 parameter 생성에 사용하지 않는다. 이 저장소에는 해당 모델 학습 코드도 넣지 않는다.
6. source audit, QASM constraint audit, parameter provenance, 데이터 hash, 재현 가능한 seed 및 실행 명령을 저장한다. 감사에 통과한 artifact만 `submit/`으로 패키징할 수 있다.

## 3. 데이터 처리

입력은 `public_train.csv`의 `x1~x8`과 `label`이다. loader는 열 이름, 유한값, binary label만 검증한다. 다음 처리는 하지 않는다.

- scaling 또는 normalization
- PCA 또는 feature reduction
- imputation
- augmentation
- polynomial, pair-product, Fourier, codebook feature 생성
- label을 이용한 feature selection

CSV는 로드 직후 `(n, 8)` NumPy 배열로 변환될 뿐 값은 변경되지 않는다. 원본 파일 SHA-256을 artifact에 기록한다.

## 4. 회로 구조

B1은 4 qubit, 8 raw feature, 16 trainable parameter를 사용한다.

```text
x1~x4 ── RY direct encoding on q0~q3
       ── CX(1→0), CX(3→2), CX(2→1)
       ── trainable RY/RZ theta_0~theta_7
x5~x8 ── RZ direct encoding on q0~q3
       ── CX(1→0), CX(3→2), CX(2→1)
       ── trainable RY/RZ theta_8~theta_15
       ── measure q0 only
```

각 data gate의 angle에는 정확히 하나의 raw feature만 들어간다. `x_i*x_j`, classical coefficient, label statistic은 들어가지 않는다. 허용 gate만 사용하며 회로에는 실제 CX가 포함된다.

## 5. 학습 과정

```text
label-independent random theta
              ↓
raw x → parameterized quantum circuit → StatevectorEstimator
                                      ↓
                              q0 measurement probability
                                      ↓
                   balanced binary cross-entropy with raw label
                                      ↓
                              COBYLA updates theta
```

StatevectorEstimator와 StatevectorSampler는 양자 회로를 로컬에서 실행하는 quantum emulator다. Estimator는 q0의 Z expectation에서 정확한 확률을 계산해 학습 비용을 줄이고, Sampler는 최대 1,024 shots로 validation 안정성을 평가한다. 둘 다 동일한 parameterized quantum circuit을 실행한다. COBYLA는 고전 optimizer이지만 별도의 예측 모델이 아니다. COBYLA가 받는 목적함수 값은 매번 현재 `theta`로 양자 회로를 실행해 얻은 확률에서 직접 계산된다.

학습 loss는 각 class의 평균 BCE를 1/2씩 더한다. 이는 balanced accuracy에 맞춘 objective 정의이며, label과 feature 관계를 미리 학습하는 별도 분류·회귀 모델이나 parameter transfer가 아니다.

결정 threshold는 label로 보정하지 않고 `0.5`로 고정했다. 규정상 threshold 보정이 허용되는지 별도 확인되기 전까지 이 보수적 정책을 유지한다.

## 6. 검증과 최종 학습의 분리

`screen`은 public train 내부의 stratified holdout에서 seed와 학습 예산을 평가하지만 제출 파일을 생성하지 않는다. 설정을 확정한 다음 `train-final`이 전체 public train을 동일한 직접 quantum loss 방식으로 새로 학습한다.

private/public test label, leaderboard feedback, 과거 B46 weight는 초기화·feature 선택·objective에 사용하지 않는다.

## 7. 자동 감사

`audit-source`는 알려진 classical predictive model import와 이름이 제출 프로젝트에 들어왔는지 검사한다. `audit-artifact`는 다음을 검사한다.

- `x_0~x_7`, `theta_0~theta_15` 이름
- 허용 gate, 2~8 qubit, depth 50 이하, two-qubit gate 및 단일 measurement
- weights 이름과 고정 threshold
- classical/surrogate/teacher model 없음
- warm start와 coefficient transfer 없음
- feature processing 없음

자동 검사는 거짓 기록을 발견하는 완전한 증명은 아니다. 따라서 `parameter_provenance.json`, training code, loss history와 데이터 hash를 함께 보존한다.

## 8. 현재 상태와 다음 단계

현재 단계의 목표는 높은 leaderboard 점수가 아니라 규정 준수 기준선을 확립하는 것이다. 테스트와 smoke training을 통과한 뒤 여러 random initialization seed를 동일한 train-only protocol로 비교할 수 있다. 다만 후보를 선택하더라도 모든 최종 `theta`는 full train에서 quantum measurement loss로 다시 학습해야 한다.

향후 회로를 확장할 때도 다음 조건을 유지한다.

- raw single-feature affine encoding
- label-independent initialization
- circuit-output loss를 통한 end-to-end parameter optimization
- classical model coefficient, prediction, residual, teacher signal의 이전 금지
- 제출 전 재현성 및 provenance 감사

## 9. 사전 등록 feature-layout 실험

순차 분할에는 데이터 기반의 특별한 근거가 없으며, RY 첫 블록과 RZ 두 번째 블록 및 q0~q3 위치가 서로 다른 causal role을 가지므로 raw feature 배치가 성능에 영향을 줄 수 있다. 고전 feature importance를 사용하지 않고 다음 네 배치만 사전에 고정해 동일 quantum training protocol로 비교한다.

- `sequential`: `(x1,x2,x3,x4)` / `(x5,x6,x7,x8)`
- `proposed_1347_2568`: `(x1,x3,x4,x7)` / `(x2,x5,x6,x8)`
- `odd_even`: `(x1,x3,x5,x7)` / `(x2,x4,x6,x8)`
- `block_swap`: `(x5,x6,x7,x8)` / `(x1,x2,x3,x4)`

슬래시 앞 tuple은 q0~q3의 RY, 뒤 tuple은 q0~q3의 RZ에 순서대로 직접 들어간다. 모든 배치는 raw feature 8개를 정확히 한 번씩 사용하며 feature product, scaling, label statistic을 만들지 않는다. 먼저 동일 seed의 paired comparison을 수행하고, 개선 배치만 독립 seed에서 확인한다.

## 10. 사용할 feature 수와 조합 탐색

추가 공지에 따라 8개 raw feature를 모두 사용할 의무는 없다. B1의 trainable block과 CX 구조는 고정하고, 선택한 raw feature에 해당하는 RY/RZ data gate만 남긴다. 사용하지 않는 feature parameter는 QASM에서 생략한다.

조합 탐색은 별도 고전 예측 모델 없이 quantum circuit loss만 사용한다. 빈 배치에서 시작해 아직 사용하지 않은 raw feature를 다음 활성 slot에 추가하고, validation BA가 높은 상위 `beam_width`개 순서만 다음 feature 수로 확장한다. 모든 후보는 동일한 label-independent random theta, 동일 train-only subset과 split, 동일 EstimatorQNN balanced BCE 예산을 사용한다.

민감도 감사에서 기존 sequential B1의 `x6~x8` slot은 q0 확률 변화가 약 `1e-12`로 사실상 비활성임을 확인했다. 따라서 feature를 원래 번호 slot에 남겨두지 않고 순서대로 `RY(q0), RY(q1), RY(q2), RY(q3), RZ(q0)`의 다섯 활성 slot에 pack한다. 1차 기본값은 600행, 25% validation, 18 COBYLA evaluations, beam width 2, feature 수 1~5이다. 이 결과는 빠른 순위 선별용이며 제출 weights를 만들지 않는다. 상위 조합은 전체 6,000행과 여러 seed에서 다시 직접 quantum training한 뒤에만 최종 후보가 될 수 있다.

### 10.1 2026-08-04 quantum-only feature 탐색 결과

1차 packed beam search는 600개 train-only 표본을 450/150으로 나누고 49개 ordered candidate를 평가했다. 피처 개수별 최고 validation BA는 1개 `x2` 0.701387, 2개 `x2,x3` 0.754963, 3개 `x2,x6,x8` 0.777804, 4개 `x2,x3,x5,x8` 0.777804, 5개 `x2,x5,x6,x7,x8` 0.737025였다. 같은 BA를 기록한 후보 중 4개 조합은 validation balanced BCE가 0.550220으로 가장 낮아 1차 우선 후보였다. 이 값은 18회 최적화의 screen 결과이므로 최종 성능으로 해석하지 않는다.

사용자 지정에 따라 2개, 3개, 5개 대표 후보만 전체 6,000행의 동일 holdout(split seed 2026, fit 4,800/validation 1,200, COBYLA 80 evaluations, 1,024 shots, threshold 0.5)에서 확인했다.

| 피처 수 | 활성 slot 순서 | fit BA | validation BA | validation BCE |
|---:|---|---:|---:|---:|
| 2 | `x2,x3` | 0.762439 | 0.747154 | 0.599927 |
| 3 | `x2,x6,x8` | 0.763006 | 0.745489 | 0.631454 |
| 5 | `x2,x5,x6,x7,x8` | 0.742522 | **0.757054** | **0.490483** |

현재 확인한 후보 중에는 5개 조합이 BA와 BCE 모두 가장 좋다. 다만 1차 최고였던 4개 조합은 이번 사용자 지정 확인에서 제외됐고, 독립 split seed 반복도 아직 수행하지 않았으므로 최종 feature set으로 확정하지 않는다.

여기서 `x6~x8`의 비활성은 피처 자체의 성질이 아니라 원래 sequential B1에서 각각 들어가던 `RZ(q1)~RZ(q3)` slot의 인과적 성질이다. packed 실험에서는 `x6~x8`도 `RY(q0)~RY(q3)` 또는 `RZ(q0)`에 재배치될 수 있으므로 실제 q0 확률에 영향을 준다. 따라서 원형 B1의 단순 gate 제거 실험과 packed ordered feature-selection 실험을 서로 같은 것으로 해석하지 않는다.

## 11. C1 causal data-reuploading VQC

B1의 causal-cone 결함과 약한 표현력을 동시에 해결하기 위해 C1을 새로 구현했다. C1은 4 qubit, depth 23, CX 12개, trainable parameter 65개를 사용한다. `(x1,x2,x3,x4)`, `(x5,x6,x7,x8)`을 이 순서로 두 번 재업로드하며 각 raw data gate는 `RY(theta_scale*x_i + theta_bias)`의 단일-feature affine 형태만 사용한다. 각 block은 trainable RZ/RY mixer 뒤에 `CX(3→2), CX(2→1), CX(1→0)`를 적용하여 먼 qubit의 정보를 먼저 q0 방향으로 전달한다.

theta는 label과 무관하게 생성한다. affine scale은 1 주변 uniform random, bias와 mixer는 0 주변 uniform random으로 초기화한다. 별도 고전 predictive/surrogate/teacher model, warm start 또는 coefficient transfer는 없다. 목적함수는 C1 q0 exact statevector probability의 balanced BCE이며 L-BFGS-B는 이 회로의 analytic adjoint gradient만 사용한다. 벡터화 statevector 확률은 Qiskit Statevector와 최대 약 `1.6e-15` 차이로 일치했다.

### 11.1 train-only 검증

1,200행 screen의 3-seed exact validation BA는 0.811111, 0.833333, 0.771111로 평균 0.805185였다. 전체 6,000행에서 80% fit/20% validation을 사용한 3-seed 결과는 다음과 같다.

| seed | fit exact BA | validation exact BA | validation 1,024-shot BA | validation balanced BCE |
|---:|---:|---:|---:|---:|
| 2026 | 0.809309 | 0.782093 | 0.788219 | 0.432762 |
| 2027 | 0.805259 | 0.816566 | 0.820458 | 0.406845 |
| 2028 | 0.806101 | 0.799873 | 0.805445 | 0.416991 |

평균 exact validation BA는 0.799511, 평균 shot BA는 0.804707이다. 모든 seed에서 x1~x8의 q0 finite-difference sensitivity가 nonzero였다.

### 11.2 5-fold OOF와 final 후보

split seed 2026, init seed 2027, fold별 80 L-BFGS-B iteration의 genuine 5-fold OOF BA는 threshold 0.5에서 0.806648이었다. C1 OOF probability만으로 선택한 threshold 0.495에서는 0.807106, 1,024-shot BA는 0.805324였다. fold별 BA는 0.792284, 0.812689, 0.810476, 0.806555, 0.811211이다.

동일 설정으로 전체 6,000행을 무작위 초기값부터 다시 학습한 final 후보의 full-train exact BA는 0.806448, shot BA는 0.805335, balanced BCE는 0.411586이다. 제출 artifact는 `artifacts/20260804T052607Z_final_c1_seed2027`, 패키지는 `submit/c1_causal_reupload_oof_seed2027`이다. artifact audit는 회로 제약, single-feature affine encoding, x1~x8 각 2회 사용, q0 단일 측정, theta 이름, threshold provenance 및 고전 사전학습 부재를 모두 통과했다.

## 12. 회로 용량 진단과 F1 eight-qubit 설계

### 12.1 기존 변형이 모두 같은 지점에서 멈춘 관찰

2026-08-06에 이미 구현되어 있던 아홉 개 변형을 동일 프로토콜(train-only 1,200행,
split seed 2026, validation 25%, 동일 초기화 seed)로 처음으로 모두 스크리닝했다.
validation exact BA는 C1O 0.7278, D1S 0.7583, C2 0.7944, D1O 0.8000, C1 0.8028,
C3 0.8083, E1S 0.8083, D1 0.8139, C1X 0.8222였다. C1/C1X/E1S를 split seed
2026~2028로 반복한 평균은 각각 0.8083, 0.8111, 0.8139로 seed 간 표준편차(약 0.03)
안에 모두 들어간다.

즉 encoding 배치나 업로드 횟수를 바꾸는 변형만으로는 0.80~0.81 부근을 벗어나지
못했다. 이 계열은 전부 큐비트 2~4개, depth 15~23, CX 12개 이하를 사용하며 규정이
허용하는 8 큐비트 / depth 50 / 2-qubit gate 80개 예산의 절반 이하만 쓴다.
따라서 병목을 학습 방식이 아니라 회로 표현력으로 판단했다.

### 12.2 C1 causal funnel의 구조적 한계

C1의 `CX(3→2), CX(2→1), CX(1→0)`는 세 게이트가 큐비트를 공유해 depth 3을 소모하면서도
한 block 안에서 q3의 정보가 q0까지 도달하지 못한다. 큐비트 수를 늘려 선형 퍼널을
연장하면 이 문제는 더 나빠진다.

### 12.3 F1: 이진 트리 퍼널

F1은 8 큐비트에서 얽힘을 이진 트리로 구성한다.

```text
depth 1: CX(1→0), CX(3→2), CX(5→4), CX(7→6)   (서로 겹치지 않음)
depth 2: CX(2→0), CX(6→4)
depth 3: CX(4→0)
```

CX 7개와 depth 3만으로 여덟 큐비트 전부가 q0의 backward causal cone에 들어온다.
block 하나는 data RY(depth 1) + RZ mixer(1) + RY mixer(1) + 트리 퍼널(3) = depth 6,
CX 7개를 쓴다. block 8개면 depth 50, CX 56개, trainable parameter 257개로 규정
상한에 정확히 맞는다.

각 data gate는 여전히 공지에서 허용한 단일 피처 affine `RY(theta_scale*x_i + theta_bias)`
뿐이며, 모든 피처 상호작용은 회로 내부의 CX에서만 생성된다. block `b`는 feature
`(q + b) % 8`을 qubit `q`에 배치해 각 피처가 업로드마다 퍼널의 다른 위치를 거치게 한다.
이 배치는 사전에 고정된 규칙이며 label이나 데이터 통계를 사용하지 않는다.

### 12.4 학습 방법론 동일성

C1에서 확립한 조건은 그대로 유지한다. theta는 label과 무관한 무작위 초기값에서
시작하고(affine scale은 1 주변, bias와 mixer는 0 주변 uniform), 목적함수는 F1 q0
exact statevector probability의 balanced BCE이며, L-BFGS-B는 이 회로의 analytic
adjoint gradient만 사용한다. 고전 predictive/surrogate/teacher model, warm start,
coefficient transfer는 없다.

### 12.5 8큐비트 시뮬레이터의 두 가지 구현 조건

상태가 C1의 16배(256 진폭)라 C1 시뮬레이터를 그대로 확장할 수 없었다.

첫째, C1은 gradient를 위해 게이트마다 직전 상태를 저장한다. 8큐비트·6,000행·249게이트에서
이는 약 6GB가 되므로, 정통 adjoint 방식으로 바꿔 최종 상태만 남기고 역방향에서
역게이트를 적용해 각 직전 상태를 복원한다. 최대 메모리는 회로 깊이와 무관하게
statevector 두 개다.

둘째, 배치 축을 맨 뒤로 옮겨(`(2,)*8 + (rows,)`) 낮은 번호 큐비트에서 stride-2
인터리브 접근이 생기지 않게 하고, 행을 64개씩 나눠 처리해 statevector가 캐시에
상주하도록 했다. 전체 6,000행 기준 gradient 1회 평가가 27초에서 6.5초로 줄었다.

두 최적화는 loss와 gradient의 값을 바꾸지 않는다. 회귀 테스트로 청킹 유무의 loss 차이
0.0, gradient 차이 6.5e-19를 확인했고, Qiskit `Statevector` 대비 확률 오차는 약
`9e-16`, adjoint gradient 대 유한차분 오차는 약 `9e-11`이다. 또한 block 하나만으로도
x1~x8 전부가 q0에 nonzero 민감도를 갖는지 테스트로 검사한다.

## 13. 측정 해상도 문제와 F1 기각

### 13.1 1,200행 screen의 노이즈 바닥

동일한 C1 설정이 split seed 2026/2027/2028에서 0.8333, 0.8361, 0.7722를 기록했다.
240행 validation에서 balanced accuracy의 표준편차는 `sqrt(0.8*0.2/120 * 2)/2 ≈ 0.026`이며,
관측된 0.056 폭은 이 노이즈만으로 완전히 설명된다.

따라서 9종 아키텍처 비교(0.79~0.82), feature layout 4종, C1/C1X/E1S 비교, F1 block 수
비교는 모두 노이즈 안에서 이루어졌으며 증거로 사용할 수 없다. 이후 모든 선택 판단은
전체 6,000행 5-fold OOF(노이즈 약 ±0.005) 또는 전체 데이터 holdout(±0.011)에서만 한다.

### 13.2 F1 기각

전체 6,000행 C1 5-fold OOF는 400 iteration에서 0.8044(threshold 0.49에서 0.8054)로,
80 iteration 기록(0.8067/0.8071)과 사실상 동일했다. 즉 C1은 이미 수렴해 있었고 반복
증가는 도움이 되지 않는다. 이 값은 실제 private leaderboard 점수 0.805와도 일치하므로,
train-only OOF는 leaderboard의 신뢰할 수 있는 예측기다.

F1은 1,200행 screen에서 block 4가 평균 0.7546, block 6이 0.7907로 C1의 0.8139를
넘지 못했다. block 8은 파라미터 257개에 fit 행 960개로 행/파라미터 비가 3.7:1이었다.
전체 데이터에서 재확인하기 전에 아래 두 결과가 나와 F1 계열 확장을 중단했다.

- 초기화 폭 실험(13.3)에서 loss 지형이 평평함이 확인되어, 표현력이 아니라 지형이
  이미 포화 상태임이 드러났다.
- 데이터 구조 조사(13.4)에서 실효 차원이 약 3으로 확인되어, 257 파라미터가
  필요하다는 전제 자체가 성립하지 않음이 드러났다.

### 13.3 초기화 폭: loss 지형은 평평하다

이 저장소의 모든 실험은 `init_scale=0.05`를 사용했다. 이 값에서는 8개의 "독립적인"
restart 초기값이 서로 L2 거리 0.42 안에 있고 초기값 자체의 norm은 4.03이다. 즉 65차원
공간에서 항상 같은 한 점 근처에서만 출발해 왔으며, multi-restart는 아무것도 탐색하지
않았다.

전체 6,000행에서 초기화 폭을 넓혀 재실험했다(각 8 restart, 200 iteration).

| init_scale / jitter | 초기값 L2 확산 | 최종 학습 loss | validation BA |
|---|---:|---:|---:|
| 0.05 / 0.1 | 0.45 | 0.40184 | 0.7848 |
| 0.50 / 0.3 | 3.05 | 0.40258 | 0.7832 |
| 1.50 / 0.6 | 8.85 | 0.40058 | 0.7932 |
| 3.14 / 1.0 | 18.34 | 0.40233 | 0.8027 |

초기값 확산이 40배 넓어져도 최종 학습 loss는 0.4006~0.4026으로 폭이 0.002뿐이다.
C1의 loss 지형에는 더 나은 분지가 존재하지 않으며, 국소최적해 가설은 기각된다.

### 13.4 데이터 구조 조사 (label 미사용)

회로 설계 근거를 확보하기 위해 x1~x8의 구조만 조사했다. label과의 관계는 보지 않았고,
데이터를 변형하지도 않았으므로 규정 1 및 공지 #5와 무관하다.

- 모든 피처가 정확히 `[-π, π]` 안에 있고 최댓값이 π와 일치한다. 주최측이 각도 인코딩에
  맞춰 스케일링해 배포했다는 뜻이며, `theta_weight ≈ 1`이 자연스러운 스케일임을 확인했다.
  기존 인코딩 기저는 적절했다.
- ±π 경계에서 밀도가 0에 수렴한다(x2 가장자리 빈 5·4 대 내부 중앙값 262). 순환 위상이
  아니라 유한 구간으로 압착된 연속 변수이므로, 주파수를 정수로 제약할 이유는 없다.
- 고유값 개수는 x1이 147개, (x2,x3,x4)가 1719개, (x5,x6,x7,x8)이 1960개다. 각 블록이
  유한한 풀에서 표집·재조합되었음을 뜻한다.
- x3는 x2의 2차 조화와, x6·x7·x8은 x5와 독립 가정 대비 20배 이상 강하게 결합돼 있다.
  실효 차원은 8이 아니라 약 3이다.
- x1은 std 0.372로 다른 피처(0.61~1.61)보다 훨씬 좁고 `|mean exp(i x1)| = 0.956`으로
  좁은 호에 집중돼 있다. 정보량이 매우 적으며, 10.1절의 quantum-only beam search가
  x1을 어떤 상위 조합에도 선택하지 않은 결과와 일치한다.

실효 차원 3에 대해 C1의 65개, F1의 257개 파라미터는 모두 충분하다. 회로 용량이 병목이
아니라는 13.2와 13.3의 결론이 데이터 쪽에서도 지지된다.

### 13.5 남은 가설: 목적함수와 채점 지표의 불일치

학습은 balanced BCE를, 채점은 threshold 기준 balanced accuracy를 사용한다. 학습된 C1
확률 분포에서 전체 행의 82%가 threshold로부터 0.1보다 멀리 있어, cross-entropy gradient의
대부분이 분류 결과를 바꿀 수 없는 영역에 쓰인다. 1,024 shot 노이즈가 threshold를 넘길 수
있는 행은 5.2%뿐이므로 shot 강건성은 병목이 아니다.

`objectives.soft_balanced_accuracy`는 하드 지시함수를 `sigmoid((p - t)/T)`로 바꾸고
온도를 낮춰 채점 지표에 직접 정렬한다. T → 0에서 실제 balanced accuracy로 수렴함을
확인했고(T=0.02에서 0.8429 대 참값 0.8437), 회로 adjoint를 통과한 gradient는 유한차분과
`9.7e-11` 이내로 일치한다. 무작위 시작점에서는 gradient가 소실되므로 balanced BCE로
워밍업한 뒤 어닐링한다. 모든 단계가 회로 확률과 raw train label만 사용한다.

## 14. G1 latent-group 회로와 과소적합 진단

### 14.1 G1 설계와 결과

13.4절의 구조 조사를 반영해 G1을 만들었다. x1을 사용하지 않고(공지 #5가 허용),
각 큐비트가 한 그룹의 모든 view를 순차적으로 받는다. 축을 RY/RZ로 번갈아 쓰는 이유는,
같은 축 회전이 연속되면 두 피처가 섞인 하나의 각도로 합성되어 단일 피처 각도 규정 위반으로
읽힐 여지가 있기 때문이다. 축을 번갈아 쓰면 각 게이트가 명확히 단일 피처 affine으로 남는다.
여러 view의 결합은 고전 연산이 아니라 회로가 수행하며, 이는 명시적으로 허용된 data
re-uploading과 같은 원리다.

blocks=4에서 depth 31, CX 12개, 파라미터 145개다. 전체 6,000행 holdout 결과는
validation BA 0.7916, 학습 loss 0.40559로 같은 프로토콜의 C1 기준선(0.7848~0.8027,
loss 0.40058~0.40258)과 구분되지 않았다. x1의 q0 민감도는 정확히 0으로 확인됐다.

또한 이 시점부터 회로를 선언적 `CircuitSpec`으로 기술하고 QASM 빌더와 시뮬레이터를 같은
명세에서 생성한다. 기존처럼 아키텍처마다 빌더와 시뮬레이터를 따로 손으로 작성하면 둘이
어긋날 수 있으며, 실제로 restart 배선 버그가 cross-validation 경로에만 남아 있었던 것도
같은 복제 구조 때문이었다.

### 14.2 과소적합 진단

C1을 소량 데이터에 과적합시킬 수 있는지 확인했다.

| 학습 행 수 | train BA |
|---:|---:|
| 20 | 1.0000 |
| 50 | 0.9400 |
| 100 | 0.9200 |
| 200 | 0.8650 |
| 500 | 0.8260 |
| 4,800 | 0.8128 |

20행은 완전히 외우지만 전체 데이터에서는 train BA가 0.813에서 멈춘다. fit 0.813 대
validation 0.792로 격차가 작으므로 과적합이 아니라 과소적합이며, 리더보드 상한 0.84에
비해 **훈련 적합 단계에서 이미 약 0.03이 부족하다.** 파라미터를 65 → 145 → 257로 늘려도
train BA가 오르지 않았으므로 파라미터 개수의 문제가 아니다.

### 14.3 인코딩 주파수는 원인이 아니다

날카로운 결정 경계에는 고주파 성분이 필요하고, 경사하강은 저주파에서 고주파로 넘어가기
어렵다(spectral bias). affine scale의 초기 중심과 경계를 올려 확인했다.

| 초기 중심 | scale 경계 | train BA | loss | 학습된 max abs scale |
|---:|---:|---:|---:|---:|
| 1 | 3 | 0.8109 | 0.4044 | 2.459 |
| 3 | 10 | 0.8073 | 0.4053 | 3.996 |
| 6 | 20 | 0.7454 | 0.5107 | 8.874 |
| 12 | 40 | 0.6746 | 0.5957 | 18.034 |

경계를 40까지 열어도 최적해의 최대 scale은 2.46에 머물며, 높은 주파수에서 시작하면
오히려 크게 나빠진다. 이 데이터가 요구하는 결정 경계는 매끄럽고 저주파다.

### 14.4 벽의 정체는 목적함수였다

같은 C1 회로와 같은 balanced BCE 워밍업에서 출발해 `soft_balanced_accuracy`를
어닐링한 결과다.

| 목적함수 | fit BA | validation BA |
|---|---:|---:|
| balanced BCE | 0.8105 | 0.7776 |
| 어닐링 soft balanced accuracy | **0.8376** | **0.7932** |

단계별 train BA는 0.8105 → 0.7895 → 0.8028 → 0.8139 → 0.8209 → 0.8320 → 0.8376이다.
첫 단계의 하락은 T=0.2가 balanced accuracy의 근사로 지나치게 매끄럽기 때문이며 이후는
단조 증가한다.

즉 0.81의 훈련 적합 한계는 회로 표현력이 아니라 balanced BCE가 선택하던 지점이었다.
BCE는 확률 보정 오차에 벌점을 주므로, balanced accuracy가 더 높지만 보정이 덜 된 해를
체계적으로 회피한다. 채점 지표는 threshold 기준 분류 결과만 보므로 두 최적해가 다르다.

### 14.5 그러나 OOF에서는 이득이 거의 사라진다

전체 6,000행 5-fold OOF(노이즈 약 ±0.005)로 확인한 결과다.

| 지표 | balanced BCE | 어닐링 soft BA |
|---|---:|---:|
| OOF BA @0.5 | 0.8044 | 0.8083 |
| OOF BA @선택 threshold | 0.8054 | 0.8096 |
| OOF shot BA @선택 threshold | 0.8030 | 0.8070 |
| 선택된 threshold | 0.4900 | 0.4575 |
| OOF balanced BCE | 0.4157 | 0.4815 |

fold별 BA @0.5는 BCE가 0.7867, 0.8116, 0.8088, 0.8032, 0.8118이고 어닐링이 0.7984,
0.8077, 0.8077, 0.8155, 0.8124로 3승 2패다. 차이의 평균은 +0.0039, 짝지은 t는 약 1.2로
유의하지 않다.

단계별 이득은 train +0.027, holdout validation +0.016, OOF +0.004로 계속 줄어든다.
어닐링이 학습 fold의 경계 근처 행을 맞추고 있으며 그 적합이 전이되지 않는다는 뜻이다.
부작용도 있다. 확률 보정이 나빠지고(OOF BCE 0.4157 → 0.4815) 선택 threshold가
0.49에서 0.4575로 이동하며, 1,024-shot 노이즈가 threshold를 넘길 수 있는 행이
5.8%에서 6.9%로 늘어난다.

따라서 목적함수 정렬은 방향은 옳지만 `temperature_stop = 0.02`는 지나치게 낮다.
정지 온도는 과적합을 조절하는 실질적 하이퍼파라미터이며, train이나 단일 holdout 수치가
아니라 OOF에서만 선택해야 한다. 13.1절에서 정한 원칙을 목적함수 실험에도 동일하게
적용한다.

또한 F1과 G1은 모두 balanced BCE 아래에서만 평가했다. BCE가 회로와 무관하게 0.81
부근을 선택한다는 것이 확인된 이상, 두 아키텍처의 기각은 아직 확정적이지 않다.

## 15. AUC 대리함수

### 15.1 설계 근거

14.5절에서 soft balanced accuracy가 실패한 원인은 명확했다. T가 낮아질수록 목적함수가
`|p - threshold| < T`인 행들에 독점되며, T=0.02에서는 6,000행 중 100여 행이 전체
gradient를 차지한다. 그 행들을 맞춘 결과가 전이되지 않았다.

`objectives.smooth_auc`는 반대 성질을 갖는다. 각 쌍의 지시함수 `1[p+ > p-]`를
`sigmoid((p+ - p-)/T)`로 바꾸고 모든 양성/음성 쌍에 평균한다. 모든 행이 gradient를
나눠 가지므로 표본이 붕괴하지 않으며, 목적함수에 threshold가 등장하지 않는다.
threshold는 학습이 끝난 뒤 out-of-fold 확률에서만 고른다.

brute force와 오차 0.0으로 일치하고, T → 0에서 순위 기반 참 AUC로 수렴하며
(T=0.005에서 0.896216 대 0.896356), 회로 adjoint를 통과한 gradient는 유한차분과
`8.6e-10` 이내로 일치한다. 6,000행의 900만 쌍 계산에 약 0.55초가 걸린다. 순위 기반
`O(n log n)` 단축은 하드 지시함수에만 적용되므로 전체 쌍을 정확히 계산하되 블록으로
나눠 메모리만 제한한다.

### 15.2 결과

split seed 2026, 워밍업과 restart와 fold와 온도 격자를 14.5절과 동일하게 맞춰 대리함수만
바꿨다.

| stage | T | OOF BA @선택 threshold | OOF shot BA |
|---:|---:|---:|---:|
| 0 | 워밍업만 | 0.8050 | 0.8042 |
| 3 | 0.1542 | 0.8077 | 0.8087 |
| 4 | 0.1105 | 0.8129 | 0.8076 |
| 5 | 0.0792 | 0.8133 | 0.8125 |
| 6 | 0.0568 | **0.8160** | 0.8148 |
| 7 | 0.0407 | 0.8140 | 0.8100 |
| 8 | 0.0292 | 0.8126 | 0.8078 |
| 9 | 0.0209 | 0.8127 | 0.8111 |
| 10 | 0.0150 | 0.8086 | 0.8072 |

| 목적함수 | 최적 OOF BA |
|---|---:|
| balanced BCE | 0.8054 |
| 어닐링 soft balanced accuracy | 0.8097 |
| 어닐링 smooth AUC | **0.8160** |

BCE 대비 +0.0110으로 OOF 노이즈 ±0.005의 약 2.2배다. 더 중요한 것은 곡선의 모양으로,
stage 4부터 9까지 여섯 구간이 모두 0.8126 이상이다. soft balanced accuracy의 좁은
봉우리와 달리 T를 보수적으로 골라도 결과가 유지되므로, 상관된 11개 지점의 최댓값을
취한 선택 편향으로는 설명되지 않는다. shot BA도 함께 따라오므로 1,024-shot 평가에서
손실이 없다.

다만 T는 여전히 OOF에서 선택했고 split seed 하나의 결과이므로, 독립 split seed
2027과 2028에서 재현을 확인한 뒤에만 제출 후보로 삼는다. 재현 확인을 위해 각 온도
단계의 fold별 balanced accuracy를 artifact에 함께 기록한다.
