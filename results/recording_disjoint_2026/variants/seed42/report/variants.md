| Recipe variant / head (stored RP configurations, pooled) | Accuracy [95 % interval] | Balanced acc. | Ternary recall | Recording-level acc. | vs stored WavKAN (same configurations) | vs stored same head (same configurations) |
|---|---|---|---|---|---|---|
| aug / WavKAN head (n=1) | 0.504 [0.468, 0.541] | 0.499 | 0.451 | 0.596 | +0.066 [+0.026, +0.107]* | +0.066 [+0.026, +0.107]* |
| aug / MLP head (n=2) | 0.557 [0.524, 0.590] | 0.549 | 0.481 | 0.676 | +0.104 [+0.069, +0.141]* | +0.065 [+0.040, +0.092]* |
| aug+views / MLP head (n=1) | 0.589 [0.552, 0.625] | 0.583 | 0.527 | 0.682 | +0.151 [+0.104, +0.197]* | +0.104 [+0.074, +0.133]* |
| bal / MLP head (n=1) | 0.478 [0.445, 0.513] | 0.470 | 0.394 | 0.522 | +0.040 [+0.004, +0.077]* | -0.007 [-0.026, +0.013] |
| bandnorm / MLP head (n=1) | 0.506 [0.473, 0.541] | 0.501 | 0.451 | 0.565 | +0.068 [+0.026, +0.110]* | +0.021 [-0.003, +0.046] |
| base / MLP head (n=2) | 0.486 [0.454, 0.517] | 0.478 | 0.409 | 0.567 | +0.033 [+0.004, +0.062]* | -0.006 [-0.019, +0.007] |
| mixup / MLP head (n=1) | 0.495 [0.462, 0.529] | 0.488 | 0.423 | 0.580 | +0.057 [+0.020, +0.094]* | +0.010 [-0.013, +0.033] |
| pool / MLP head (n=1) | 0.483 [0.450, 0.517] | 0.474 | 0.396 | 0.549 | +0.045 [+0.006, +0.084]* | -0.002 [-0.024, +0.019] |
| valloss / MLP head (n=1) | 0.469 [0.432, 0.508] | 0.461 | 0.390 | 0.502 | +0.031 [-0.009, +0.071] | -0.016 [-0.042, +0.010] |
| views / MLP head (n=2) | 0.521 [0.485, 0.558] | 0.511 | 0.412 | 0.565 | +0.068 [+0.035, +0.103]* | +0.029 [+0.011, +0.048]* |

| Other inputs, per method | Accuracy [95 % interval] | Balanced acc. | Ternary recall | Recording-level acc. | vs stored RP + WavKAN (mean of the RP configurations) | vs stored RP, same head (mean), or logistic regression on RP |
|---|---|---|---|---|---|---|
| logistic regression on RP | 0.441 [0.415, 0.467] | 0.437 | 0.392 | 0.514 | -0.018 [-0.040, +0.007] | +0.000 [-0.002, +0.002] |
| logistic regression on RP, class-balanced | 0.440 [0.413, 0.466] | 0.437 | 0.405 | 0.510 | -0.020 [-0.042, +0.005] | -0.002 [-0.005, +0.002] |
| logistic regression on RP, band-normalized | 0.443 [0.416, 0.471] | 0.440 | 0.411 | 0.576 | -0.016 [-0.040, +0.007] | +0.002 [-0.015, +0.019] |
| logistic regression on RP without fluctuation weighting | 0.440 [0.413, 0.465] | 0.435 | 0.392 | 0.514 | -0.020 [-0.042, +0.005] | -0.002 [-0.004, +0.001] |
| logistic regression on RH | 0.473 [0.437, 0.509] | 0.463 | 0.371 | 0.518 | +0.014 [-0.013, +0.040] | +0.032 [+0.004, +0.059]* |
| logistic regression on TRH | 0.473 [0.437, 0.509] | 0.463 | 0.371 | 0.518 | +0.014 [-0.013, +0.041] | +0.032 [+0.004, +0.059]* |
| logistic regression on tempogram-ratio statistics (26) | 0.638 [0.598, 0.676] | 0.643 | 0.689 | 0.745 | +0.179 [+0.142, +0.217]* | +0.197 [+0.160, +0.234]* |
| logistic regression on onset scale features (32) | 0.623 [0.588, 0.660] | 0.622 | 0.604 | 0.780 | +0.164 [+0.126, +0.204]* | +0.182 [+0.145, +0.220]* |
| logistic regression on RH + TRH + tempogram-ratio statistics (26) + onset scale features (32) | 0.632 [0.594, 0.668] | 0.635 | 0.668 | 0.737 | +0.172 [+0.137, +0.206]* | +0.190 [+0.158, +0.222]* |
| logistic regression on TSSD | 0.415 [0.384, 0.448] | 0.405 | 0.306 | 0.455 | -0.044 [-0.081, -0.008]* | -0.026 [-0.062, +0.008] |
| logistic regression on RP + RH + TRH + TSSD | 0.456 [0.426, 0.485] | 0.451 | 0.410 | 0.557 | -0.004 [-0.027, +0.019] | +0.014 [-0.002, +0.031] |
| logistic regression on RP + RH + TRH + TSSD + tempogram-ratio statistics (26) + onset scale features (32) | 0.526 [0.494, 0.557] | 0.524 | 0.509 | 0.631 | +0.066 [+0.039, +0.093]* | +0.084 [+0.064, +0.106]* |
| logistic regression on tempogram-ratio means (13) | 0.631 [0.591, 0.669] | 0.635 | 0.668 | 0.737 | +0.172 [+0.133, +0.211]* | +0.190 [+0.152, +0.227]* |
| logistic regression on tempogram-ratio deviations (13) | 0.505 [0.469, 0.539] | 0.489 | 0.337 | 0.588 | +0.045 [+0.014, +0.076]* | +0.064 [+0.030, +0.097]* |
| logistic regression on tempogram-ratio quantiles (39) | 0.624 [0.586, 0.659] | 0.627 | 0.655 | 0.733 | +0.165 [+0.129, +0.202]* | +0.183 [+0.148, +0.220]* |
| logistic regression on tempogram-ratio statistics (26) + onset scale features (32) | 0.664 [0.628, 0.697] | 0.667 | 0.691 | 0.808 | +0.204 [+0.167, +0.241]* | +0.222 [+0.187, +0.257]* |
| logistic regression on tempogram-ratio statistics (26), class-balanced | 0.635 [0.592, 0.673] | 0.647 | 0.763 | 0.722 | +0.175 [+0.134, +0.215]* | +0.193 [+0.155, +0.232]* |
| logistic regression on tempogram-ratio statistics (26) + onset scale features (32), class-balanced | 0.663 [0.626, 0.698] | 0.672 | 0.749 | 0.796 | +0.204 [+0.164, +0.242]* | +0.222 [+0.185, +0.257]* |
| logistic regression on tempogram-ratio statistics (26) + onset scale features (32) + RH | 0.632 [0.594, 0.668] | 0.635 | 0.668 | 0.733 | +0.172 [+0.138, +0.206]* | +0.190 [+0.158, +0.222]* |
| ratio network, 16 ch, 5x5, WavKAN head, base | 0.608 [0.573, 0.641] | 0.612 | 0.647 | 0.725 | +0.149 [+0.114, +0.184]* | +0.149 [+0.114, +0.184]* |
| ratio network, 16 ch, 5x5, MLP head, base | 0.627 [0.590, 0.660] | 0.631 | 0.668 | 0.733 | +0.167 [+0.131, +0.203]* | +0.139 [+0.105, +0.174]* |
| ratio network, 32 ch, 7x7, WavKAN head, base | 0.618 [0.581, 0.657] | 0.621 | 0.649 | 0.702 | +0.159 [+0.123, +0.195]* | +0.159 [+0.123, +0.195]* |
| ratio network, 32 ch, 7x7, MLP head, base | 0.636 [0.599, 0.674] | 0.638 | 0.660 | 0.722 | +0.177 [+0.141, +0.212]* | +0.149 [+0.113, +0.186]* |
| WavKAN head on tempogram-ratio statistics + onset scale features (58), base | 0.565 [0.528, 0.598] | 0.558 | 0.494 | 0.671 | +0.105 [+0.067, +0.143]* | +0.105 [+0.067, +0.143]* |
| MLP head on tempogram-ratio statistics + onset scale features (58), base | 0.642 [0.604, 0.679] | 0.646 | 0.687 | 0.757 | +0.183 [+0.144, +0.221]* | +0.155 [+0.118, +0.190]* |
| linear head on tempogram-ratio statistics + onset scale features (58), base | 0.617 [0.578, 0.656] | 0.618 | 0.627 | 0.698 | +0.158 [+0.117, +0.200]* | +0.170 [+0.132, +0.208]* |
| WavKAN head on tempogram-ratio statistics (26), base | 0.501 [0.465, 0.537] | 0.490 | 0.392 | 0.584 | +0.041 [+0.005, +0.078]* | +0.041 [+0.005, +0.078]* |
| MLP head on tempogram-ratio statistics (26), base | 0.622 [0.582, 0.657] | 0.624 | 0.651 | 0.718 | +0.162 [+0.127, +0.199]* | +0.134 [+0.101, +0.169]* |
| linear head on tempogram-ratio statistics (26), base | 0.582 [0.541, 0.621] | 0.579 | 0.546 | 0.655 | +0.123 [+0.084, +0.162]* | +0.135 [+0.098, +0.171]* |
| WavKAN head on onset scale features (32), base | 0.560 [0.524, 0.593] | 0.554 | 0.502 | 0.702 | +0.100 [+0.064, +0.139]* | +0.100 [+0.064, +0.139]* |
| MLP head on onset scale features (32), base | 0.628 [0.590, 0.666] | 0.630 | 0.654 | 0.773 | +0.168 [+0.128, +0.208]* | +0.141 [+0.102, +0.179]* |
| linear head on onset scale features (32), base | 0.610 [0.572, 0.652] | 0.606 | 0.562 | 0.710 | +0.151 [+0.112, +0.192]* | +0.163 [+0.124, +0.202]* |
| WavKAN head on tempogram-ratio statistics + onset scale features (58), bal | 0.559 [0.522, 0.593] | 0.563 | 0.604 | 0.643 | +0.099 [+0.059, +0.139]* | +0.099 [+0.059, +0.139]* |
| MLP head on tempogram-ratio statistics + onset scale features (58), bal | 0.633 [0.594, 0.670] | 0.641 | 0.720 | 0.749 | +0.173 [+0.130, +0.214]* | +0.146 [+0.107, +0.184]* |
| linear head on tempogram-ratio statistics + onset scale features (58), bal | 0.621 [0.581, 0.661] | 0.630 | 0.714 | 0.698 | +0.161 [+0.118, +0.205]* | +0.173 [+0.133, +0.213]* |
| WavKAN head on tempogram-ratio statistics + onset scale features (58), valloss | 0.541 [0.504, 0.577] | 0.537 | 0.492 | 0.627 | +0.082 [+0.042, +0.123]* | +0.082 [+0.042, +0.123]* |
| MLP head on tempogram-ratio statistics + onset scale features (58), valloss | 0.616 [0.575, 0.656] | 0.618 | 0.637 | 0.710 | +0.156 [+0.116, +0.196]* | +0.129 [+0.090, +0.168]* |
| linear head on tempogram-ratio statistics + onset scale features (58), valloss | 0.608 [0.566, 0.649] | 0.608 | 0.608 | 0.686 | +0.148 [+0.105, +0.191]* | +0.160 [+0.120, +0.200]* |
| WavKAN head on tempogram-ratio statistics + onset scale features (58), ls | 0.557 [0.521, 0.591] | 0.550 | 0.482 | 0.671 | +0.097 [+0.059, +0.136]* | +0.097 [+0.059, +0.136]* |
| MLP head on tempogram-ratio statistics + onset scale features (58), ls | 0.640 [0.603, 0.677] | 0.644 | 0.676 | 0.761 | +0.181 [+0.143, +0.220]* | +0.153 [+0.117, +0.188]* |
| linear head on tempogram-ratio statistics + onset scale features (58), ls | 0.632 [0.593, 0.670] | 0.632 | 0.629 | 0.733 | +0.173 [+0.132, +0.213]* | +0.184 [+0.147, +0.222]* |
| WavKAN head on tempogram-ratio statistics + onset scale features (58), ens5 | 0.573 [0.536, 0.608] | 0.567 | 0.509 | 0.686 | +0.113 [+0.076, +0.152]* | +0.113 [+0.076, +0.152]* |
| MLP head on tempogram-ratio statistics + onset scale features (58), ens5 | 0.643 [0.603, 0.680] | 0.647 | 0.682 | 0.769 | +0.183 [+0.145, +0.222]* | +0.156 [+0.119, +0.192]* |
| linear head on tempogram-ratio statistics + onset scale features (58), ens5 | 0.628 [0.589, 0.668] | 0.629 | 0.635 | 0.725 | +0.169 [+0.128, +0.210]* | +0.181 [+0.143, +0.220]* |

`*`: the 95 % paired cluster-bootstrap interval of the difference excludes zero.
