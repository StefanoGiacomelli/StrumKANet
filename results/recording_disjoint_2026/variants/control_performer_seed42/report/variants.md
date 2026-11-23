| Recipe variant / head (stored RP configurations, pooled) | Accuracy [95 % interval] | Balanced acc. | Ternary recall | Recording-level acc. | vs stored WavKAN (same configurations) | vs stored same head (same configurations) |
|---|---|---|---|---|---|---|

| Other inputs, per method | Accuracy [95 % interval] | Balanced acc. | Ternary recall | Recording-level acc. | vs stored RP + WavKAN (mean of the RP configurations) | vs stored RP, same head (mean), or logistic regression on RP |
|---|---|---|---|---|---|---|
| logistic regression on RP | 0.433 [0.405, 0.458] | 0.428 | 0.385 | 0.561 | -0.023 [-0.046, -0.001]* | -0.001 [-0.002, +0.001] |
| logistic regression on tempogram-ratio statistics (26) | 0.643 [0.604, 0.680] | 0.647 | 0.684 | 0.753 | +0.188 [+0.146, +0.230]* | +0.210 [+0.171, +0.248]* |
| logistic regression on onset scale features (32) | 0.612 [0.575, 0.649] | 0.611 | 0.598 | 0.745 | +0.156 [+0.116, +0.199]* | +0.179 [+0.140, +0.217]* |
| logistic regression on tempogram-ratio statistics (26) + onset scale features (32) | 0.653 [0.616, 0.690] | 0.657 | 0.694 | 0.784 | +0.198 [+0.157, +0.240]* | +0.220 [+0.182, +0.259]* |

`*`: the 95 % paired cluster-bootstrap interval of the difference excludes zero.
