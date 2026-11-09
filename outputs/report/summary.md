| Métrica | S5-nano (Maritaca) | Gemma 4 E2B | MiniCPM-V 4.7 1B | Nemotron 3.5 Bolt E4B |
|---|---|---|---|---|
| Validade estrutural | 0.366 → 0.997 | 0.783 → 0.997 | 0.000 → 0.992 | 0.134 → 0.997 |
| Perplexidade (ponderada por token) | 4.498 → 1.995 | 11.372 → 2.212 | 19.151 → 4.016 | 9.078 → 1.999 |
| NLL média da referência | 1.507 → 0.689 | 2.432 → 0.790 | 2.961 → 1.382 | 2.213 → 0.691 |
| Similaridade com o conceito | 0.471 → 0.537 | 0.500 → 0.544 | 0.479 → 0.547 | 0.558 → 0.540 |
| Não redundância semântica | 0.265 → 0.238 | 0.188 → 0.211 | 0.000 → 0.099 | 0.227 → 0.237 |
| Vendi score @5 | 2.915 → 2.728 | 2.480 → 2.611 | 1.000 → 2.111 | 2.684 → 2.726 |
| Alinhamento com a referência | 0.524 → 0.644 | 0.520 → 0.629 | 0.270 → 0.571 | 0.564 → 0.646 |
| Sobreposição de emoji (diagnóstico) | 0.268 → 0.344 | 0.272 → 0.341 | 0.000 → 0.228 | 0.252 → 0.364 |
| Latência média (s) | 1.755 → 4.443 | 1.798 → 4.818 | 0.842 → 3.369 | 2.261 → 6.435 |
| Tokens gerados | 76.925 → 110.832 | 72.976 → 105.840 | 54.426 → 113.868 | 67.395 → 105.758 |

Métricas semânticas: média sobre todas as saídas válidas de cada variante; os deltas pareados estão em summary.json.

Modelos adaptados, par a par (esquerda − direita), conceitos válidos nos dois; IC 95% por bootstrap pareado.

| Par | n | Alinhamento | Similaridade | Não redundância | Vendi @5 |
|---|---|---|---|---|---|
| s5-nano − gemma4-e2b | 2187 | +0.015 [+0.012, +0.018] | -0.007 [-0.010, -0.004] | +0.028 [+0.023, +0.032] | +0.118 [+0.097, +0.139] |
| s5-nano − minicpm-v4.7-1b | 2176 | +0.074 [+0.070, +0.078] | -0.010 [-0.013, -0.006] | +0.140 [+0.134, +0.145] | +0.619 [+0.591, +0.647] |
| s5-nano − nemotron-bolt-e4b | 2187 | -0.002 [-0.005, +0.002] | -0.003 [-0.006, +0.000] | +0.001 [-0.003, +0.006] | +0.003 [-0.017, +0.022] |
| gemma4-e2b − minicpm-v4.7-1b | 2175 | +0.059 [+0.056, +0.063] | -0.002 [-0.006, +0.001] | +0.111 [+0.105, +0.117] | +0.496 [+0.468, +0.525] |
| gemma4-e2b − nemotron-bolt-e4b | 2186 | -0.017 [-0.020, -0.013] | +0.004 [+0.001, +0.008] | -0.026 [-0.031, -0.021] | -0.114 [-0.135, -0.093] |
| minicpm-v4.7-1b − nemotron-bolt-e4b | 2177 | -0.076 [-0.080, -0.072] | +0.007 [+0.003, +0.010] | -0.138 [-0.143, -0.132] | -0.611 [-0.639, -0.582] |

Saídas com texto curto repetido: reference 0.037 (n=2199), s5-nano 0.219 (n=2193), gemma4-e2b 0.300 (n=2192), minicpm-v4.7-1b 0.642 (n=2182), nemotron-bolt-e4b 0.259 (n=2193)
