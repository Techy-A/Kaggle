# genfix bench

molecules 450, errors 0, with list 450

identity: GENFIX=False vs original base-list cell on 16 molecules: byte-identical True (differing [], original re-run differing None); main-loop base list == identity-run list: 16/16; offline genfix rebuild == notebook GENFIX=True list: 16/16; notebook genfix stats {'runs': 16, 'failed': 0, 'new': 144, 'inserted': 144, 'inserted_top25': 66, 'inserted_top1': 0}

engine-1 CPU time per molecule: run 1 4.4s, gen20 run 6.7s (genfix = 2.53x engine 1); generated candidates 18 -> 51; candidates 128 -> 161
score scale: top-25 scores sd 2.74, median gap rank1-rank2 2.12

## ungated gen20 (one engine-1 run, list-relative features re-scored)

| slice | n | base MRR | var MRR | d | top-1 base->var | correct loses rank 1 | gains rank 1 | top-5 order of originals changed | new cand at rank 1 | mean new in top 25 |
|---|---|---|---|---|---|---|---|---|---|---|
| thin c1 (np) | 70 | 0.8901 | 0.8217 | -0.0685 | 58->50 | 8 | 0 | 0.557 | 0.029 | 4.11 |
| thin c1 (t) | 200 | 0.9178 | 0.8912 | -0.0267 | 174->165 | 10 | 1 | 0.285 | 0.045 | 3.72 |
| c1 all | 270 | 0.9107 | 0.8731 | -0.0375 | 232->215 | 18 | 1 | 0.356 | 0.041 | 3.82 |
| c2 | 50 | 0.9513 | 0.9511 | -0.0003 | 46->46 | 0 | 0 | 0.480 | 0.000 | 4.18 |
| c3 | 130 | 0.3144 | 0.4307 | +0.1163 | 38->51 | 0 | 13 | 0.546 | 0.108 | 3.90 |

paired bootstrap: natural mix d +0.0111 [-0.0068, +0.0303] P(better) 0.89 P(worse) 0.11; 28/20/52 mix d +0.0499 [+0.0238, +0.0799] P(better) 1.00 P(worse) 0.00
  class 1: d -0.0375 [-0.0555, -0.0214] P(better) 0.00
  class 2: d -0.0003 [-0.0008, +0.0000] P(better) 0.00
  class 3: d +0.1163 [+0.0686, +0.1719] P(better) 1.00

## genfix penalty sweep (all molecules; selection is in-sample here)

| delta | c1 MRR d | c1 rank-1 losses | c2 d | c3 d | c3 gains rank 1 | mix d |
|---|---|---|---|---|---|---|
| -1 | -0.0390 | 17 | -0.0005 | +0.1191 | 13 | +0.0509 |
| -0.5 | -0.0265 | 11 | -0.0003 | +0.1179 | 13 | +0.0538 |
| 0 | -0.0202 | 9 | +0.0000 | +0.1136 | 12 | +0.0534 |
| 0.25 | -0.0183 | 8 | +0.0000 | +0.1131 | 12 | +0.0537 |
| 0.5 | -0.0130 | 6 | +0.0000 | +0.1129 | 12 | +0.0551 |
| 0.75 | -0.0111 | 5 | +0.0000 | +0.1072 | 11 | +0.0526 |
| 1 | -0.0062 | 3 | +0.0000 | +0.0994 | 9 | +0.0500 |
| 1.5 | -0.0006 | 0 | +0.0000 | +0.0928 | 8 | +0.0481 |
| 2 | -0.0006 | 0 | +0.0000 | +0.0828 | 6 | +0.0429 |
| 3 | +0.0000 | 0 | +0.0000 | +0.0625 | 3 | +0.0325 |
| 5 | +0.0000 | 0 | +0.0000 | +0.0277 | 0 | +0.0144 |

cross-fit penalties: half 0 scored with delta 1.5 (chosen on half 1), half 1 with delta 1.0

## genfix, cross-fitted penalty

| slice | n | base MRR | var MRR | d | top-1 base->var | correct loses rank 1 | gains rank 1 | top-5 order of originals changed | new cand at rank 1 | mean new in top 25 |
|---|---|---|---|---|---|---|---|---|---|---|
| thin c1 (np) | 70 | 0.8901 | 0.8901 | +0.0000 | 58->58 | 0 | 0 | 0.000 | 0.000 | 2.50 |
| thin c1 (t) | 200 | 0.9178 | 0.9095 | -0.0083 | 174->171 | 3 | 0 | 0.000 | 0.020 | 2.48 |
| c1 all | 270 | 0.9107 | 0.9045 | -0.0062 | 232->229 | 3 | 0 | 0.000 | 0.015 | 2.49 |
| c2 | 50 | 0.9513 | 0.9513 | +0.0000 | 46->46 | 0 | 0 | 0.000 | 0.000 | 2.76 |
| c3 | 130 | 0.3144 | 0.4093 | +0.0949 | 38->46 | 0 | 8 | 0.000 | 0.069 | 2.31 |

paired bootstrap: natural mix d +0.0237 [+0.0111, +0.0384] P(better) 1.00 P(worse) 0.00; 28/20/52 mix d +0.0476 [+0.0257, +0.0734] P(better) 1.00 P(worse) 0.00
  class 1: d -0.0062 [-0.0136, -0.0006] P(better) 0.00
  class 2: d +0.0000 [+0.0000, +0.0000] P(better) 0.00
  class 3: d +0.0949 [+0.0543, +0.1415] P(better) 1.00

## genfix at delta 1.5 (chosen on all molecules: in-sample)

| slice | n | base MRR | var MRR | d | top-1 base->var | correct loses rank 1 | gains rank 1 | top-5 order of originals changed | new cand at rank 1 | mean new in top 25 |
|---|---|---|---|---|---|---|---|---|---|---|
| thin c1 (np) | 70 | 0.8901 | 0.8901 | +0.0000 | 58->58 | 0 | 0 | 0.000 | 0.000 | 2.29 |
| thin c1 (t) | 200 | 0.9178 | 0.9170 | -0.0008 | 174->174 | 0 | 0 | 0.000 | 0.005 | 2.25 |
| c1 all | 270 | 0.9107 | 0.9100 | -0.0006 | 232->232 | 0 | 0 | 0.000 | 0.004 | 2.26 |
| c2 | 50 | 0.9513 | 0.9513 | +0.0000 | 46->46 | 0 | 0 | 0.000 | 0.000 | 2.58 |
| c3 | 130 | 0.3144 | 0.4072 | +0.0928 | 38->46 | 0 | 8 | 0.000 | 0.069 | 2.14 |

paired bootstrap: natural mix d +0.0265 [+0.0145, +0.0405] P(better) 1.00 P(worse) 0.00; 28/20/52 mix d +0.0481 [+0.0268, +0.0736] P(better) 1.00 P(worse) 0.00
  class 1: d -0.0006 [-0.0019, +0.0000] P(better) 0.00
  class 2: d +0.0000 [+0.0000, +0.0000] P(better) 0.00
  class 3: d +0.0928 [+0.0524, +0.1392] P(better) 1.00
lib_max [0, 0.5): n 85, mean new in top 25 2.05, d MRR +0.0126
lib_max [0.5, 0.7): n 97, mean new in top 25 2.80, d MRR +0.0653
lib_max [0.7, 0.9): n 228, mean new in top 25 2.21, d MRR +0.0184
lib_max [0.9, 2): n 40, mean new in top 25 1.73, d MRR +0.0075
