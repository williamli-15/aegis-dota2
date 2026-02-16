# Phase 1 Report

## Value Model (TEST patch)

| Model | AUC | LogLoss | RMSE_dGold | MAE_dGold | RMSE_dXP | MAE_dXP | n_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Value v0 | 0.7699 | 0.5633 | 2753.9432 | 1881.4940 | 5011.4459 | 3172.6296 | 147611 |
| Value v1 | 0.7827 | 0.5508 | 2280.0780 | 1572.9653 | 4266.9917 | 2713.0549 | 147611 |

## Next-item Policy (TEST)

| Top1 | Top5 | Top20 | classes | n_rows |
| --- | --- | --- | --- | --- |
| 0.2190 | 0.5518 | 0.8890 | 80 | 260878 |

## Generate-and-Score (TEST)

### Magic toggle

| policy_hit@k | k | avg_uplift_vs_base | gs_hit@1_true_ok | avg_uplift_vs_true_true_ok | n_true_ok | n_eval |
| --- | --- | --- | --- | --- | --- | --- |
| 0.8872 | 20 | 0.0183 | 0.0464 | 0.0165 | 5000 | 5000 |

### Feasible/time-band

| policy_hit@k | k | avg_uplift_vs_base | gs_hit@1_true_ok | avg_uplift_vs_true_true_ok | n_true_ok | n_eval |
| --- | --- | --- | --- | --- | --- | --- |
| 0.8872 | 20 | 0.0097 | 0.1226 | 0.0093 | 1998 | 5000 |
