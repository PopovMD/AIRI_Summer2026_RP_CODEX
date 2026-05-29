# DEG-weighted Gaussian NLL patch for sschrod/CODEX

This patch modifies the Norman/GEARS gene perturbation workflow:

- `CODEX_gene_perturbation.py`
  - adds CLI flags for DEG weighting;
  - enables `--deg_weighting` by default;
  - appends the DEG-weighting parameters to `experiment_description` so runs are distinguishable.

- `codex/reconstruction_utils.py`
  - builds a condition-level DEG weight matrix from `adata.uns['rank_genes_groups_cov']`, which is assigned to `adata.uns['top_non_zero_de_20']` in the original loader;
  - computes effect sizes from training perturbation means versus training control means;
  - adds a per-cell condition id to the treated `TensorDataset` when DEG weighting is enabled.

- `codex/CODEX_reconstruction.py`
  - replaces unweighted `nn.GaussianNLLLoss()` in training with a normalized weighted Gaussian NLL:
    `sum(weight * nll) / sum(weight)`;
  - keeps the original behavior when `--no-deg_weighting` is passed.

Weight formula used for condition `c` and ranked DEG `g`:

```text
w(c, g) = 1 + min(deg_weight_max,
                  deg_weight_alpha * |mean_c(g) - mean_ctrl(g)| / deg_effect_scale)
              * clip((deg_top_k - rank(g)) / deg_top_k, 0, 1)
```

All non-DEG genes keep weight `1.0`.

Example Norman run:

```bash
docker run -it --rm --gpus "device=0" -v /sybig/home/ssc/CODEX:/mnt codex \
  python3 -i CODEX_gene_perturbation.py \
  -l 512 128 64 -s 1 -dn norman \
  --deg_weighting --deg_weight_alpha 1.0 --deg_weight_max 5.0 --deg_effect_scale 1.0 --deg_top_k 20
```

To recover the original unweighted training behavior:

```bash
python3 -i CODEX_gene_perturbation.py -l 512 128 64 -s 1 -dn norman --no-deg_weighting
```
