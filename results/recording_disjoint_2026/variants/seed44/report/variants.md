| Recipe variant / head (stored RP configurations, pooled) | Accuracy [95 % interval] | Balanced acc. | Ternary recall | Recording-level acc. | vs stored WavKAN (same configurations) | vs stored same head (same configurations) |
|---|---|---|---|---|---|---|
| aug / WavKAN head (n=1) | 0.496 [0.464, 0.528] | 0.489 | 0.427 | 0.596 | +0.065 [+0.021, +0.107]* | +0.065 [+0.021, +0.107]* |
| aug / MLP head (n=1) | 0.534 [0.502, 0.567] | 0.527 | 0.462 | 0.678 | +0.103 [+0.062, +0.143]* | - |
| views / MLP head (n=1) | 0.507 [0.467, 0.549] | 0.507 | 0.510 | 0.565 | +0.075 [+0.034, +0.114]* | - |

| Other inputs, per method | Accuracy [95 % interval] | Balanced acc. | Ternary recall | Recording-level acc. | vs stored RP + WavKAN (mean of the RP configurations) | vs stored RP, same head (mean), or logistic regression on RP |
|---|---|---|---|---|---|---|
| logistic regression on RP | 0.422 [0.395, 0.450] | 0.419 | 0.392 | 0.490 | -0.029 [-0.055, -0.002]* | +0.000 [-0.002, +0.002] |
| logistic regression on tempogram-ratio statistics (26) | 0.628 [0.586, 0.666] | 0.632 | 0.670 | 0.729 | +0.176 [+0.131, +0.220]* | +0.206 [+0.165, +0.245]* |
| logistic regression on onset scale features (32) | 0.615 [0.578, 0.650] | 0.613 | 0.592 | 0.741 | +0.164 [+0.122, +0.208]* | +0.193 [+0.154, +0.233]* |
| logistic regression on tempogram-ratio statistics (26) + onset scale features (32) | 0.656 [0.617, 0.692] | 0.659 | 0.687 | 0.788 | +0.205 [+0.162, +0.249]* | +0.234 [+0.196, +0.272]* |
| logistic regression on tempogram-ratio statistics (26), class-balanced | 0.628 [0.586, 0.669] | 0.640 | 0.752 | 0.718 | +0.177 [+0.131, +0.223]* | +0.206 [+0.164, +0.248]* |
| ratio network, 16 ch, 5x5, WavKAN head, base | 0.599 [0.560, 0.636] | 0.601 | 0.623 | 0.694 | +0.148 [+0.104, +0.192]* | +0.148 [+0.104, +0.192]* |
| MLP head on tempogram-ratio statistics + onset scale features (58), base | 0.628 [0.589, 0.666] | 0.632 | 0.676 | 0.753 | +0.176 [+0.132, +0.223]* | - |
| MLP head on tempogram-ratio statistics + onset scale features (58), bal | 0.629 [0.591, 0.666] | 0.637 | 0.717 | 0.741 | +0.178 [+0.132, +0.223]* | - |
| WavKAN head on tempogram-ratio statistics + onset scale features (58), base | 0.555 [0.519, 0.592] | 0.549 | 0.494 | 0.643 | +0.104 [+0.060, +0.146]* | +0.104 [+0.060, +0.146]* |
| WavKAN head on tempogram-ratio statistics + onset scale features (58), bal | 0.549 [0.512, 0.585] | 0.554 | 0.604 | 0.647 | +0.098 [+0.052, +0.143]* | +0.098 [+0.052, +0.143]* |
| linear head on tempogram-ratio statistics + onset scale features (58), base | 0.623 [0.583, 0.662] | 0.622 | 0.612 | 0.749 | +0.172 [+0.127, +0.217]* | - |
| linear head on tempogram-ratio statistics + onset scale features (58), bal | 0.618 [0.579, 0.656] | 0.625 | 0.691 | 0.729 | +0.167 [+0.122, +0.215]* | - |

`*`: the 95 % paired cluster-bootstrap interval of the difference excludes zero.
