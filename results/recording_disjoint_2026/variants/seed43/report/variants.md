| Recipe variant / head (stored RP configurations, pooled) | Accuracy [95 % interval] | Balanced acc. | Ternary recall | Recording-level acc. | vs stored WavKAN (same configurations) | vs stored same head (same configurations) |
|---|---|---|---|---|---|---|
| aug / WavKAN head (n=1) | 0.502 [0.466, 0.537] | 0.490 | 0.380 | 0.576 | +0.029 [-0.002, +0.062] | +0.029 [-0.002, +0.062] |
| aug / MLP head (n=1) | 0.537 [0.506, 0.569] | 0.527 | 0.433 | 0.651 | +0.064 [+0.029, +0.099]* | - |
| views / MLP head (n=1) | 0.509 [0.471, 0.550] | 0.508 | 0.498 | 0.573 | +0.037 [+0.005, +0.067]* | - |

| Other inputs, per method | Accuracy [95 % interval] | Balanced acc. | Ternary recall | Recording-level acc. | vs stored RP + WavKAN (mean of the RP configurations) | vs stored RP, same head (mean), or logistic regression on RP |
|---|---|---|---|---|---|---|
| logistic regression on RP | 0.435 [0.409, 0.462] | 0.431 | 0.396 | 0.510 | -0.049 [-0.076, -0.022]* | +0.000 [-0.002, +0.002] |
| logistic regression on tempogram-ratio statistics (26) | 0.630 [0.591, 0.666] | 0.635 | 0.681 | 0.725 | +0.146 [+0.107, +0.186]* | +0.195 [+0.155, +0.233]* |
| logistic regression on onset scale features (32) | 0.620 [0.585, 0.657] | 0.619 | 0.610 | 0.757 | +0.136 [+0.098, +0.177]* | +0.185 [+0.147, +0.222]* |
| logistic regression on tempogram-ratio statistics (26) + onset scale features (32) | 0.655 [0.618, 0.693] | 0.659 | 0.694 | 0.792 | +0.171 [+0.131, +0.212]* | +0.220 [+0.183, +0.257]* |
| logistic regression on tempogram-ratio statistics (26), class-balanced | 0.629 [0.590, 0.668] | 0.642 | 0.762 | 0.710 | +0.145 [+0.105, +0.188]* | +0.194 [+0.154, +0.233]* |
| ratio network, 16 ch, 5x5, WavKAN head, base | 0.587 [0.549, 0.624] | 0.586 | 0.578 | 0.702 | +0.103 [+0.063, +0.142]* | +0.103 [+0.063, +0.142]* |
| MLP head on tempogram-ratio statistics + onset scale features (58), base | 0.626 [0.590, 0.663] | 0.631 | 0.683 | 0.761 | +0.142 [+0.100, +0.183]* | - |
| MLP head on tempogram-ratio statistics + onset scale features (58), bal | 0.621 [0.583, 0.660] | 0.632 | 0.734 | 0.741 | +0.137 [+0.095, +0.180]* | - |
| WavKAN head on tempogram-ratio statistics + onset scale features (58), base | 0.569 [0.535, 0.602] | 0.566 | 0.540 | 0.667 | +0.085 [+0.045, +0.126]* | +0.085 [+0.045, +0.126]* |
| WavKAN head on tempogram-ratio statistics + onset scale features (58), bal | 0.565 [0.529, 0.599] | 0.569 | 0.608 | 0.686 | +0.081 [+0.041, +0.122]* | +0.081 [+0.041, +0.122]* |
| linear head on tempogram-ratio statistics + onset scale features (58), base | 0.612 [0.573, 0.652] | 0.614 | 0.636 | 0.706 | +0.128 [+0.088, +0.171]* | - |
| linear head on tempogram-ratio statistics + onset scale features (58), bal | 0.598 [0.558, 0.639] | 0.609 | 0.713 | 0.675 | +0.114 [+0.073, +0.157]* | - |

`*`: the 95 % paired cluster-bootstrap interval of the difference excludes zero.
