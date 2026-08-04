# Quantum AI Competition — clean restart

이 저장소는 대회 추가 공지의 classical pre-training 금지 규정을 기준으로 처음부터 다시 만든 제출용 프로젝트다.

핵심 원칙은 하나다. 모든 `theta_N`은 무작위 초기값에서 시작하며, raw feature를 입력받은 양자 회로의 정확한 statevector 기대확률로 계산한 balanced cross-entropy만을 이용해 갱신한다. Logistic Regression, SVM, QDA, tree/boosting, neural-network teacher, ridge codebook 또는 그 계수는 제출 파라미터 생성에 사용하지 않는다. 1,024-shot SamplerQNN은 validation과 최종 안정성 평가에 사용한다.

## 준비

`raw/public_train.csv`를 둔다. CSV는 `x1`~`x8`, `label` 열을 가져야 하며 이 코드는 scaling, PCA, imputation, augmentation을 수행하지 않는다.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## 실행

빠른 규정·동작 검사:

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m qchallenge.cli audit-source
```

train-only holdout screening:

```powershell
.venv\Scripts\python.exe -m qchallenge.cli screen --data-dir raw --seed 2026 --maxiter 80 --shots 1024
```

사전 등록한 raw feature 배치를 동일 조건으로 비교할 때는 `--feature-layout`에
`sequential`, `proposed_1347_2568`, `odd_even`, `block_swap` 중 하나를 지정한다.
어떤 배치도 feature 값을 변환하거나 둘 이상을 한 gate angle에 결합하지 않는다.

사용할 raw feature의 개수와 조합을 저비용 quantum-only beam search로 찾는다.

```powershell
.venv\Scripts\python.exe -m qchallenge.cli search-features --data-dir raw --screen-rows 600 --maxiter 18 --beam-width 2 --max-features 5
```

이 명령은 제출 파일을 만들지 않는다. 동일한 raw train subset, split, random theta와
EstimatorQNN loss로 후보를 비교하며 결과는 `feature_search_metrics.json`에만 기록한다.
선택한 일부 feature로 정식 screen을 할 때는 예를 들어
`--features 1 3 4 7 --pack-features`를 사용한다. 이 경우 입력 순서대로
`RY(q0), RY(q1), RY(q2), RY(q3)`에 배치한다.

선택한 설정을 전체 train에 직접 학습:

```powershell
.venv\Scripts\python.exe -m qchallenge.cli train-final --data-dir raw --seed 2026 --maxiter 160 --shots 1024
```

생성된 artifact 감사 및 제출 패키징:

```powershell
.venv\Scripts\python.exe -m qchallenge.cli audit-artifact --artifact-dir artifacts/<run_id>
.venv\Scripts\python.exe -m qchallenge.cli package --artifact-dir artifacts/<run_id> --name compliant_b1
```

상세한 규정 판단과 파라미터 생성 흐름은 `docs/compliance_restart_report_ko.md`에 기록한다.

## C1 causal data-reuploading VQC

C1은 4 qubit에서 raw feature 8개를 두 번 직접 재업로드한다. 각 data gate는 공지에서 허용한 단일 피처 affine 각도 `theta_scale*x_i + theta_bias`만 사용하며, 모든 상호작용은 회로 내부의 `CX(3→2), CX(2→1), CX(1→0)` causal funnel에서 생성한다. 학습기는 동일 회로의 정확한 statevector q0 확률과 adjoint gradient를 계산하며 Qiskit Statevector와 `1e-10` 이내 동치성을 검사한다.

```powershell
.venv\Scripts\python.exe -m qchallenge.cli screen-c1 --data-dir raw --maxiter 80 --shots 1024
.venv\Scripts\python.exe -m qchallenge.cli cross-validate-c1 --data-dir raw --folds 5 --split-seed 2026 --init-seed 2027 --maxiter 80
.venv\Scripts\python.exe -m qchallenge.cli train-final-c1 --data-dir raw --seed 2027 --maxiter 80 --threshold 0.495
```

`cross-validate-c1`의 threshold는 별도 고전 모델이 아니라 각 행을 보지 않은 C1 quantum circuit의 OOF 확률만으로 선택한다.
