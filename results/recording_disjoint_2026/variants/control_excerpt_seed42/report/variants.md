| Recipe variant / head (stored RP configurations, pooled) | Accuracy [95 % interval] | Balanced acc. | Ternary recall | Recording-level acc. | vs stored WavKAN (same configurations) | vs stored same head (same configurations) |
|---|---|---|---|---|---|---|

| Other inputs, per method | Accuracy [95 % interval] | Balanced acc. | Ternary recall | Recording-level acc. | vs stored RP + WavKAN (mean of the RP configurations) | vs stored RP, same head (mean), or logistic regression on RP |
|---|---|---|---|---|---|---|
| logistic regression on RP | 0.593 [0.573, 0.614] | 0.588 | 0.546 | 0.886 | -0.073 [-0.088, -0.058]* | +0.000 [-0.002, +0.002] |
| logistic regression on tempogram-ratio statistics (26) | 0.665 [0.627, 0.700] | 0.669 | 0.711 | 0.776 | -0.001 [-0.036, +0.033] | +0.072 [+0.035, +0.108]* |
| logistic regression on onset scale features (32) | 0.665 [0.631, 0.699] | 0.663 | 0.643 | 0.808 | -0.001 [-0.033, +0.032] | +0.072 [+0.039, +0.107]* |
| logistic regression on tempogram-ratio statistics (26) + onset scale features (32) | 0.717 [0.684, 0.748] | 0.719 | 0.739 | 0.859 | +0.051 [+0.020, +0.080]* | +0.124 [+0.092, +0.155]* |

`*`: the 95 % paired cluster-bootstrap interval of the difference excludes zero.
