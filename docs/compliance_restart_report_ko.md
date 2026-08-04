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
4. 모든 `theta_0~theta_15`는 `SamplerQNN` 측정확률과 train label 사이의 binary cross-entropy만으로 COBYLA가 갱신한다.
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
raw x → parameterized quantum circuit → StatevectorSampler
                                      ↓
                              q0 measurement probability
                                      ↓
                       binary cross-entropy with raw label
                                      ↓
                              COBYLA updates theta
```

StatevectorSampler는 양자 회로를 로컬에서 실행하는 quantum emulator다. COBYLA는 고전 optimizer이지만 별도의 예측 모델이 아니다. COBYLA가 받는 목적함수 값은 매번 현재 `theta`로 양자 회로를 실행해 얻은 측정확률에서 직접 계산된다.

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

