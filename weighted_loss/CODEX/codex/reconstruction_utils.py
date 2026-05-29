import numpy as np
import pandas as pd
import scanpy as sc
import torch
from gears import PertData
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def _dense_mean(x):
    if hasattr(x, "todense"):
        return np.asarray(x.mean(axis=0)).ravel()
    return np.asarray(x).mean(axis=0).ravel()


def _build_deg_weight_matrix(args, adata, treated_train, control_train):
    """Build condition-level DEG weights for use inside the training loss.

    For each perturbation condition c and ranked DEG g:
        w(c, g) = 1 + min(max_extra, alpha * |mean_c(g)-mean_ctrl(g)| / effect_scale)
                  * clip((top_k - rank(g)) / top_k, 0, 1)

    Non-DEG genes keep weight 1.0. The returned matrix has shape
    [n_train_conditions, n_genes] and is indexed by the condition ids stored in
    the treated TensorDataset.
    """
    if not args.get("deg_weighting", False):
        args["deg_weight_matrix"] = None
        args["deg_condition_to_id"] = None
        return None, None

    rank_genes_groups_cov = adata.uns.get("rank_genes_groups_cov", {})
    if not isinstance(rank_genes_groups_cov, dict):
        raise ValueError("Expected adata.uns['rank_genes_groups_cov'] to be a dict: condition -> DEG gene names.")

    top_k = int(args.get("deg_top_k", 20))
    alpha = float(args.get("deg_weight_alpha", 1.0))
    max_extra = float(args.get("deg_weight_max", 5.0))
    effect_scale = max(float(args.get("deg_effect_scale", 1.0)), 1e-8)

    conditions = pd.Index(treated_train.obs["pert_categories"].astype(str).unique())
    condition_to_id = {condition: i for i, condition in enumerate(conditions)}
    gene_to_idx = {gene: i for i, gene in enumerate(adata.var_names.astype(str))}

    weights = np.ones((len(conditions), adata.n_vars), dtype=np.float32)
    control_mean = _dense_mean(control_train.X)

    for condition, condition_id in condition_to_id.items():
        deg_names = list(rank_genes_groups_cov.get(str(condition), []))[:top_k]
        if len(deg_names) == 0:
            continue

        idx = treated_train.obs["pert_categories"].astype(str).values == str(condition)
        if np.sum(idx) == 0:
            continue

        condition_mean = _dense_mean(treated_train[idx].X)
        gene_indices = [gene_to_idx[g] for g in deg_names if g in gene_to_idx]
        if len(gene_indices) == 0:
            continue

        abs_delta = np.abs(condition_mean[gene_indices] - control_mean[gene_indices])
        rank = np.arange(len(gene_indices), dtype=np.float32)
        rank_gate = np.clip((float(top_k) - rank) / float(top_k), 0.0, 1.0)
        extra = np.minimum(max_extra, alpha * abs_delta / effect_scale)
        weights[condition_id, gene_indices] = 1.0 + extra * rank_gate

    args["deg_weight_matrix"] = torch.tensor(weights, dtype=torch.float32, device=device)
    args["deg_condition_to_id"] = condition_to_id
    print(
        "Built DEG weight matrix:",
        f"conditions={weights.shape[0]}, genes={weights.shape[1]},",
        f"min={weights.min():.4f}, mean={weights.mean():.4f}, max={weights.max():.4f}",
    )
    return weights, condition_to_id


def load_Combosciplex_data(args=None):
    adata = sc.read("/mnt/data/Combosciplex.h5ad")
    unique_treatments = pd.concat([adata.obs["Drug1"], adata.obs["Drug2"]]).unique()
    unique_treatments = np.append(["DMSO"], unique_treatments[unique_treatments != "DMSO"])
    num_treatments = unique_treatments[unique_treatments != "DMSO"].shape[0]
    # NOTE: treatment 0 is control.

    ood_data = adata[adata.obs["ood"]]
    train_data = adata[~adata.obs["ood"]]
    idx_train, idx_test = train_test_split(train_data.obs_names, test_size=0.25, random_state=42)
    test_data = train_data[idx_test]
    train_data = train_data[idx_train]
    control_train = train_data[train_data.obs["control"]]
    train_data = train_data[~train_data.obs["control"]]
    control_test = test_data[test_data.obs["control"]]
    test_data = test_data[~test_data.obs["control"]]
    control_test = torch.Tensor(control_test.X.todense()).to(device)

    if args is not None:
        control_train_tds = TensorDataset(
            torch.Tensor(control_train.X.todense()).to(device),
            torch.Tensor(control_train.obs[["Drug1_numeric", "Drug2_numeric"]].to_numpy()).to(device),
        )
        control_train_dl = DataLoader(control_train_tds, batch_size=args["batch_size"], shuffle=True, drop_last=True)

        train_data_tds = TensorDataset(
            torch.Tensor(train_data.X.todense()).to(device),
            torch.Tensor(train_data.obs[["Drug1_numeric", "Drug2_numeric"]].to_numpy()).to(device),
        )
        train_dl = DataLoader(train_data_tds, batch_size=args["batch_size"], shuffle=True, drop_last=True)
        return train_dl, control_train_dl, control_test, test_data, ood_data, num_treatments

    return control_test, test_data, ood_data, num_treatments


def load_data(args, data_name="norman"):
    data_path = args["data_path"]
    pert_data = PertData(data_path)

    if args["download_data"]:
        pert_data.load(data_name=data_name)

    pert_data.load(data_path=data_path + data_name)
    pert_data.prepare_split(split=args["experiment"], seed=args["seed"])
    # NOTE: pert_data.set2conditions holds the train/val/test splits.
    # NOTE: pert_data.subgroup holds test subgroup splits.

    adata = pert_data.adata

    # Change uns to top 20 non-zero DE used in GEARS.
    adata.uns["rank_genes_groups_cov"] = adata.uns["top_non_zero_de_20"]
    adata.obs["pert_categories"] = adata.obs["condition_name"]
    adata.obs["Drug1"] = "ctrl"
    adata.obs["Drug2"] = "ctrl"
    adata.obs["split"] = "train"
    adata.obs["subgroup"] = "train"

    drug_list = []
    for obs_name, condition in adata.obs["condition"].items():
        if condition != "ctrl":
            condition_split = condition.split("+")
            drug_list.append(condition_split[0])
            drug_list.append(condition_split[1])
            adata.obs.loc[obs_name, "Drug1"] = condition_split[0]
            adata.obs.loc[obs_name, "Drug2"] = condition_split[1]

        if condition in pert_data.set2conditions["test"]:
            adata.obs.loc[obs_name, "split"] = "test"
            if condition in pert_data.subgroup["test_subgroup"]["combo_seen0"]:
                adata.obs.loc[obs_name, "subgroup"] = "combo_seen0"
            elif condition in pert_data.subgroup["test_subgroup"]["combo_seen1"]:
                adata.obs.loc[obs_name, "subgroup"] = "combo_seen1"
            elif condition in pert_data.subgroup["test_subgroup"]["combo_seen2"]:
                adata.obs.loc[obs_name, "subgroup"] = "combo_seen2"
            elif condition in pert_data.subgroup["test_subgroup"]["unseen_single"]:
                adata.obs.loc[obs_name, "subgroup"] = "unseen_single"

    train_data = adata[adata.obs["split"] != "test"]
    # NOTE: We use a random train/validation split, not the one provided by GEARS.
    # Unique treatments only for train data: test data might contain additional treatments.
    unique_treatments = pd.concat([train_data.obs["Drug1"], train_data.obs["Drug2"]]).unique()
    unique_treatments = np.append(["ctrl"], unique_treatments[unique_treatments != "ctrl"])
    num_treatments = unique_treatments[unique_treatments != "ctrl"].shape[0]
    # NOTE: treatment 0 is control.

    enc = LabelEncoder()
    enc.classes_ = unique_treatments
    train_data.obs["Drug1_numeric"] = enc.transform(train_data.obs["Drug1"])
    train_data.obs["Drug2_numeric"] = enc.transform(train_data.obs["Drug2"])

    idx_train, idx_test = train_test_split(train_data.obs_names, test_size=0.25, random_state=42)
    test_data = train_data[idx_test]
    train_data = train_data[idx_train]

    control_train = train_data[train_data.obs["control"] == 1]
    train_data = train_data[train_data.obs["control"] == 0]
    control_test = test_data[test_data.obs["control"] == 1]
    test_data = test_data[test_data.obs["control"] == 0]

    _, condition_to_id = _build_deg_weight_matrix(args, adata, train_data, control_train)

    control_test = torch.Tensor(control_test.X.todense()).to(device)
    control_train_tds = TensorDataset(
        torch.Tensor(control_train.X.todense()).to(device),
        torch.Tensor(control_train.obs[["Drug1_numeric", "Drug2_numeric"]].to_numpy()).to(device),
    )
    control_train_dl = DataLoader(control_train_tds, batch_size=args["batch_size"], shuffle=True, drop_last=True)

    train_tensors = [
        torch.Tensor(train_data.X.todense()).to(device),
        torch.Tensor(train_data.obs[["Drug1_numeric", "Drug2_numeric"]].to_numpy()).to(device),
    ]
    if args.get("deg_weighting", False):
        condition_ids = train_data.obs["pert_categories"].astype(str).map(condition_to_id).to_numpy()
        train_tensors.append(torch.LongTensor(condition_ids).to(device))

    train_data_tds = TensorDataset(*train_tensors)
    train_dl = DataLoader(train_data_tds, batch_size=args["batch_size"], shuffle=True, drop_last=True)

    return train_dl, control_train_dl, control_test, test_data, adata, num_treatments, pert_data, enc.classes_
