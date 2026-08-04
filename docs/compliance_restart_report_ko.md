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
