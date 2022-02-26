# Sharpness-Aware Minimization (SAM)

This repository is the unofficial implementation of [Sharpness-aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412) (ICLR 2021).
The official code can be found [here](https://github.com/google-research/sam).

## Command Lines

This projet is built on [giung2-jax](https://github.com/cs-giung/giung2-jax).
```
ln -s /path/to/giung2-jax/giung2 ./
ln -s /path/to/giung2-jax/datasets ./
```

### Train Models

Run the following command lines to train models with SAM:
```
python scripts/train_sam.py \
    --config_file ./configs/{DATASET_NAME}_{NETWORK_NAME}.yaml \
    --num_epochs {EPOCH} --num_warmup_epochs 5 \
    --batch_size 128 --learning_rate 0.1 --weight_decay 5e-4 \
    --sam_rho {RHO}
    --seed 42 --output_dir ./outputs/{DATASET_NAME}_{NETWORK_NAME}/SAM/e{EPOCH}_rho{RHO}_s42
```

### Evaluate Models

Run the following command lines to evaluate models:
```
python scripts/eval.py \
    --config_file ./configs/{DATASET_NAME}_{NETWORK_NAME}.yaml \
    --weight_file ./outputs/path/to/best_acc1
```

## Results

Refer to [`scripts/logs/README.md`](./scripts/logs/README.md) to view full results of the experiments.

### CIFAR-10 (R20-BN-ReLU)
| Epoch | ρ    | Train ACC / NLL / cNLL | Valid ACC / NLL / cNLL | Test ACC / NLL / cNLL  |
| :-:   | :-:  | :-:                    | :-:                    | :-:                    |
| 200   | 0.00 | 99.91 / 0.007 / 0.031  | 92.98 / 0.272 / 0.224  | 92.49 / 0.274 / 0.229  |
|       | 0.20 | 98.12 / 0.088 / 0.071  | 93.48 / 0.197 / 0.191  | 92.83 / 0.209 / 0.204  |
| 400   | 0.00 | 99.98 / 0.004 / 0.022  | 93.34 / 0.266 / 0.220  | 92.79 / 0.273 / 0.228  |
|       | 0.20 | 98.72 / 0.066 / 0.053  | 93.92 / 0.186 / 0.182  | 93.59 / 0.200 / 0.196  |
| 600   | 0.00 | 99.98 / 0.003 / 0.020  | 93.56 / 0.259 / 0.214  | 92.55 / 0.284 / 0.235  |
|       | 0.20 | 98.92 / 0.061 / 0.048  | 93.98 / 0.181 / 0.176  | 93.58 / 0.191 / 0.187  |
| 800   | 0.00 | 99.97 / 0.003 / 0.020  | 93.08 / 0.274 / 0.228  | 92.57 / 0.287 / 0.239  |
|       | 0.20 | 98.91 / 0.062 / 0.049  | 94.04 / 0.186 / 0.182  | 93.35 / 0.195 / 0.191  |

### CIFAR-100 (R20-BN-ReLU)
| Epoch | ρ    | Train ACC / NLL / cNLL | Valid ACC / NLL / cNLL | Test ACC / NLL / cNLL  |
| :-:   | :-:  | :-:                    | :-:                    | :-:                    |
| 200   | 0.00 | 92.42 / 0.274 / 0.385  | 68.84 / 1.210 / 1.121  | 68.22 / 1.228 / 1.136  |
|       | 0.20 | 85.80 / 0.537 / 0.528  | 69.36 / 1.040 / 1.039  | 69.51 / 1.030 / 1.030  |
| 400   | 0.00 | 95.70 / 0.175 / 0.306  | 68.40 / 1.270 / 1.144  | 68.15 / 1.276 / 1.147  |
|       | 0.20 | 88.62 / 0.454 / 0.455  | 70.18 / 1.036 / 1.036  | 70.77 / 1.010 / 1.010  |
| 600   | 0.00 | 96.97 / 0.136 / 0.275  | 67.82 / 1.315 / 1.166  | 68.04 / 1.313 / 1.163  |
|       | 0.20 | 89.64 / 0.424 / 0.427  | 70.10 / 1.020 / 1.019  | 70.66 / 1.009 / 1.009  |
| 800   | 0.00 | 97.83 / 0.111 / 0.255  | 68.24 / 1.341 / 1.170  | 67.98 / 1.345 / 1.177  |
|       | 0.20 | 90.38 / 0.402 / 0.407  | 70.76 / 1.011 / 1.011  | 70.65 / 1.010 / 1.010  |

### CIFAR-10 (WRN28x10-BN-ReLU)
| Epoch | ρ    | Train ACC / NLL / cNLL | Valid ACC / NLL / cNLL | Test ACC / NLL / cNLL  |
| :-:   | :-:  | :-:                    | :-:                    | :-:                    |
| 200   | 0.00 | 100.0 / 0.000 / 0.010  | 96.18 / 0.177 / 0.146  | 96.08 / 0.170 / 0.142  |
|       | 0.20 | 99.79 / 0.006 / 0.009  | 97.14 / 0.101 / 0.096  | 97.08 / 0.097 / 0.094  |
| 400   | 0.00 | 100.0 / 0.001 / 0.012  | 96.16 / 0.180 / 0.151  | 96.18 / 0.180 / 0.152  |
|       | 0.20 | 99.88 / 0.004 / 0.008  | 97.10 / 0.103 / 0.096  | 97.24 / 0.098 / 0.094  |

### CIFAR-100 (WRN28x10-BN-ReLU)
| Epoch | ρ    | Train ACC / NLL / cNLL | Valid ACC / NLL / cNLL | Test ACC / NLL / cNLL  |
| :-:   | :-:  | :-:                    | :-:                    | :-:                    |
| 200   | 0.00 | 99.98 / 0.001 / 0.004  | 80.04 / 0.900 / 0.844  | 80.63 / 0.843 / 0.802  |
|       | 0.40 | 99.24 / 0.029 / 0.031  | 83.62 / 0.583 / 0.582  | 83.65 / 0.574 / 0.573  |
| 400   | 0.00 | 99.99 / 0.001 / 0.004  | 79.76 / 0.906 / 0.869  | 79.95 / 0.891 / 0.857  |
|       | 0.40 | 99.02 / 0.031 / 0.033  | 83.50 / 0.602 / 0.599  | 83.50 / 0.586 / 0.584  |

### TinyImageNet-200 (R18-BN-ReLU)
| Epoch | ρ    | Train ACC / NLL / cNLL | Valid ACC / NLL / cNLL | Test ACC / NLL / cNLL  |
| :-:   | :-:  | :-:                    | :-:                    | :-:                    |
| 100   | 0.00 | 99.98 / 0.005 / 0.022  | 66.07 / 1.544 / 1.459  | 65.52 / 1.581 / 1.486  |
|       | 0.40 | 92.03 / 0.394 / 0.351  | 69.46 / 1.212 / 1.201  | 68.51 / 1.250 / 1.243  |
| 200   | 0.00 | 99.98 / 0.002 / 0.007  | 66.57 / 1.529 / 1.489  | 65.12 / 1.585 / 1.535  |
|       | 0.40 | 97.01 / 0.191 / 0.170  | 69.99 / 1.185 / 1.181  | 69.62 / 1.213 / 1.210  |
| 400   | 0.00 | 99.98 / 0.001 / 0.002  | 66.52 / 1.562 / 1.555  | 65.95 / 1.593 / 1.583  |
|       | 0.40 | 98.76 / 0.100 / 0.092  | 70.38 / 1.185 / 1.183  | 69.66 / 1.232 / 1.232  |

## License

[The MIT License](./LICENSE).
