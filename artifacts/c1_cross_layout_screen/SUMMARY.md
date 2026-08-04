# C1 alternative second-upload layout screen

## Decision

Keep the original C1 layout as the primary candidate. None of the three
alternative layouts exceeded the fair C1 reference mean balanced accuracy of
`0.807407` for initialization seed 2026. Therefore the planned 3 x 3 expansion
was not run.

## Fixed screening protocol

- Raw `public_train.csv` features only; no scaling, PCA, augmentation, or classical model.
- C1 topology fixed: 4 qubits, depth 23, 12 CX gates, q0 measurement.
- Every data gate is `RY(theta_scale * x_i + theta_bias)` with one raw feature.
- All eight features are used exactly twice.
- Balanced BCE of the exact q0 quantum probability is optimized directly with L-BFGS-B and an analytic adjoint gradient.
- Initialization seed: 2026.
- Split seeds: 2026, 2027, 2028.
- Stratified sample: 1,200 rows; validation fraction: 25%.
- Maximum optimizer iterations: 60; shot diagnostic: 1,024 shots.
- Selection metric: mean exact validation balanced accuracy at threshold 0.5.

## Results

| Layout | Split 2026 | Split 2027 | Split 2028 | Mean exact BA | Population std | Mean shot BA | Mean balanced BCE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original C1 reference | - | - | - | **0.807407** | - | - | - |
| Rotate | 0.788889 | 0.857778 | 0.766667 | **0.804444** | 0.038788 | 0.802222 | 0.434004 |
| Zip | 0.755556 | 0.866667 | 0.762222 | 0.794815 | 0.050880 | 0.796296 | 0.437190 |
| Swap | 0.791111 | 0.844444 | 0.742222 | 0.792593 | 0.041745 | 0.784444 | **0.433398** |

The high single-split Zip result (`0.866667` at split 2027) is not treated as
evidence of improvement because its other two splits are much lower and its
mean is below the reference. This avoids selecting a layout from a lucky split.

## Layouts

Feature indices below are CSV names and each tuple maps left-to-right to
qubits q0, q1, q2, q3.

- Rotate: `(x1,x2,x3,x4)`, `(x5,x6,x7,x8)`, `(x4,x1,x2,x3)`, `(x8,x5,x6,x7)`.
- Zip: `(x1,x2,x3,x4)`, `(x5,x6,x7,x8)`, `(x1,x5,x2,x6)`, `(x3,x7,x4,x8)`.
- Swap: `(x1,x2,x3,x4)`, `(x5,x6,x7,x8)`, `(x5,x6,x7,x8)`, `(x1,x2,x3,x4)`.

## Verification

All three candidates passed:

- competition circuit constraints and allowed-gate checks;
- single-feature affine-angle inspection;
- two uses of every raw feature;
- exact simulator versus Qiskit statevector equivalence (maximum error below `8e-16`);
- analytic adjoint gradient versus finite differences (maximum error below `3e-10`);
- nonzero influence of every feature on the q0 readout at initialization.

