# FEMaster benchmark suite

This repository contains the FEMaster benchmark input decks and their compressed reference results. Each model directory contains a `model.inp` input deck and a `model.res.ref.gz` reference result. Generated solver results and logs are intentionally not tracked.

The inventory below was generated with FEMaster v2.7.33 on 29 August 2026.

## Model overview

| Model | Analysis | Nodes | Elements | DOFs | Constraint method | Constraints | NNZ | Author |
|---|---|---:|---:|---:|---|---:|---:|---|
| `bracket_master` | Linear static | 168,892 | 169,167 | 1,013,352 | Null space | 864 | 54,729,504 | Felix (Come2World) |
| `bracket_master_2` | Linear static | 73,712 | 73,729 | 442,272 | Null space | 875 | 23,842,441 | Felix (Come2World) |
| `connector_1_beam` | Linear static | 1,572 | 6,822 | 4,722 | Null space | 516 | 172,458 | Finn Eggers (Luecx) |
| `connector_1_hinge` | Linear static | 1,572 | 6,822 | 4,722 | Null space | 695 | 173,257 | Finn Eggers (Luecx) |
| `connector_2` | Linear static | 604 | 2,523 | 1,812 | Null space | 84 | 69,516 | Finn Eggers (Luecx) |
| `connector_2_quad` | Linear static | 3,964 | 5,279 | 11,892 | Null space | 21 | 1,007,163 | Finn Eggers (Luecx) |
| `connector_2_test` | Linear static | 913 | 1,230 | 2,736 | Null space | 126 | 162,288 | Finn Eggers (Luecx) |
| `ecke_c3d20` | Linear static | 47,652 | 10,250 | 142,944 | Null space | 363 | 22,410,441 | Finn Eggers (Luecx) |
| `ecke_c3d20r` | Linear static | 47,652 | 30,750 | 142,944 | Null space | 363 | 22,410,441 | Finn Eggers (Luecx) |
| `haken` | Linear static | 35,809 | 15,328 | 107,427 | Null space | 831 | 16,709,562 | Finn Eggers (Luecx) |
| `shell_solid_tie` | Linear static | 673 | 784 | 2,694 | Null space | 360 | 135,324 | Finn Eggers (Luecx) |
| `solid_bracket` | Linear static | 109,613 | 66,138 | 328,839 | Null space | 612 | 24,866,991 | Sergio Pluchinsky |
| `solid_rim_m1` | Linear static | 74,468 | 40,257 | 223,407 | Elimination | 11,691 | 15,658,280 | Sergio Pluchinsky |
| `solid_rim_m2` | Linear static | 69,716 | 37,901 | 209,151 | Null space | 7,908 | 14,575,053 | Sergio Pluchinsky |
| `solid_rim_m3` | Linear static | 952 | 244 | 2,859 | Null space | 561 | 252,954 | Sergio Pluchinsky |
| `sym_hypel_rubber` | Linear static | 920 | 684 | 2,760 | Null space | 732 | 132,616 | Sergio Pluchinsky |
| `topo_bridge_1` | Linear static | 33,936 | 30,000 | 101,808 | Null space | 24 | 7,600,032 | Finn Eggers (Luecx) |
| **Total (17 models)** |  | **672,620** | **497,908** | **2,746,341** |  | **26,626** | **204,908,321** |  |

## Running the suite

From this repository, run:

```text
python3 _scripts/run.py /path/to/FEMaster
```

The runner executes every case with one and four threads, extracts structural metadata from the input and solver log, and compares the generated native result fields against the compressed reference result.
