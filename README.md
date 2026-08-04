# Quantum AI Competition — clean restart

이 저장소는 대회 추가 공지의 classical pre-training 금지 규정을 기준으로 처음부터 다시 만든 제출용 프로젝트다.

핵심 원칙은 하나다. 모든 `theta_N`은 무작위 초기값에서 시작하며, raw feature를 입력받은 양자 회로의 측정 확률로 계산한 loss만을 이용해 갱신한다. Logistic Regression, SVM, QDA, tree/boosting, neural-network teacher, ridge codebook 또는 그 계수는 제출 파라미터 생성에 사용하지 않는다.

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

