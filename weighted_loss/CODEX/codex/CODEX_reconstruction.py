import copy
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from torch import Tensor, nn

from codex.Network_base import network_block, single_layer


class CODEXReconstruction(nn.Module):
    def __init__(self, in_features, num_nodes, num_treatments=None, batch_norm=False, dropout=None, random_seed=42):
        super().__init__()
        self.num_treatments = num_treatments

        torch.manual_seed(random_seed)
        self.encoder = network_block(
            in_features,
            num_nodes[0:1],
            num_nodes[2],
            batch_norm=batch_norm,
            dropout=dropout,
            activation=nn.ReLU,
            output_activation=nn.ReLU,
            output_batch_n=batch_norm,
            output_dropout=True,
        )

        torch.manual_seed(random_seed * 2)
        self.latent_rep_dim = num_nodes[2]
        self.T_rep = torch.nn.ModuleList()
        for _ in range(num_treatments):
            net = single_layer(
                num_nodes[2],
                num_nodes[2],
                batch_norm=batch_norm,
                dropout=dropout,
                activation=nn.ReLU,
            )
            self.T_rep.append(net)

        num_nodes_flip = np.flip(num_nodes)
        torch.manual_seed(random_seed * 2)
        self.decoder = network_block(
            num_nodes_flip[0],
            num_nodes_flip[1:],
            in_features * 2,
            batch_norm=batch_norm,
            dropout=dropout,
            activation=nn.ReLU,
            output_activation=None,
            output_batch_n=False,
            output_dropout=False,
        )

    def forward(self, input: Tensor, treatment: Tensor):
        latent_rep = torch.zeros((input.shape[0], self.latent_rep_dim), device=input.device)
        dim = input.size()[1]
        embedding = self.encoder(input)

        for t in range(self.num_treatments):
            # 0 is reserved for control.
            mask = torch.any(treatment == t + 1, dim=1, keepdim=False)
            if torch.sum(mask) > 1:
                latent_rep[mask] += self.T_rep[t](embedding[mask])

        gene_reconstruction = self.decoder(latent_rep)
        # Convert variance estimates to positive values in [1e-3, infinity).
        gene_means = gene_reconstruction[:, :dim]
        gene_vars = nn.functional.softplus(gene_reconstruction[:, dim:]).add(1e-3)
        return torch.cat([gene_means, gene_vars], dim=1)

    def predict_with_weighted_perturbations(self, input: Tensor, treatment: Tensor, weight: Tensor):
        self.eval()
        latent_rep = torch.zeros((input.shape[0], self.latent_rep_dim), device=input.device)
        dim = input.size()[1]
        embedding = self.encoder(input)

        for t in range(self.num_treatments):
            # 0 is reserved for control.
            mask = torch.any(treatment == t + 1, dim=1, keepdim=False)
            if torch.sum(mask) > 1:
                latent_rep[mask] += weight[t + 1] * self.T_rep[t](embedding[mask])

        gene_reconstruction = self.decoder(latent_rep)
        gene_means = gene_reconstruction[:, :dim]
        gene_vars = nn.functional.softplus(gene_reconstruction[:, dim:]).add(1e-3)
        gene_reconstructions2 = torch.cat([gene_means, gene_vars], dim=1)
        self.train()
        return gene_reconstructions2

    def predict(self, X, treatment):
        self.eval()
        pred0 = self(X, treatment)
        self.train()
        return pred0

    def predict_numpy(self, X, treatment):
        X = torch.Tensor(X)
        out = self.predict(X, treatment)
        return out[0].detach().numpy()


def weighted_gaussian_nll_loss(input: Tensor, target: Tensor, var: Tensor, weight: Tensor | None = None, eps: float = 1e-6):
    """Gaussian NLL with optional element-wise DEG weights.

    This matches torch.nn.GaussianNLLLoss(full=False) when weight is None,
    except that the weighted case uses sum(loss * weight) / sum(weight) so
    the global loss scale remains comparable across batches.
    """
    var = var.clamp_min(eps)
    loss = 0.5 * (torch.log(var) + (input - target) ** 2 / var)

    if weight is None:
        return loss.mean()

    weight = weight.to(device=loss.device, dtype=loss.dtype)
    if weight.shape != loss.shape:
        raise ValueError(f"DEG weight tensor has shape {weight.shape}, expected {loss.shape}.")

    return (loss * weight).sum() / weight.sum().clamp_min(eps)


def _batch_deg_weights(args, batch, dim: int, device):
    if not args.get("deg_weighting", False):
        return None
    if len(batch) < 3:
        return None

    deg_weight_matrix = args.get("deg_weight_matrix")
    if deg_weight_matrix is None:
        return None

    condition_ids = batch[2].long().to(device)
    deg_weight_matrix = deg_weight_matrix.to(device)
    return deg_weight_matrix[condition_ids, :dim]


def evaluate_r2_v2(autoencoder, dataset, genes_control, for_plot=False):
    """Evaluate R2 for means and variances, including DE genes."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mean_score, var_score, mean_score_de, var_score_de = [], [], [], []
    num, dim = genes_control.size(0), genes_control.size(1)
    full_results = []
    pert_categories = []

    for pert_category in np.unique(dataset.obs.pert_categories):
        pert_categories.append(pert_category)
        idx = dataset.obs.pert_categories == pert_category
        de_idx = np.where(dataset.var_names.isin(np.array(dataset.uns["rank_genes_groups_cov"][str(pert_category)])))[0]

        if np.sum(idx) > 30:
            emb_drugs = torch.Tensor(
                dataset[idx].obs[["Drug1_numeric", "Drug2_numeric"]].to_numpy()[0:1].repeat(num, 0)
            ).to(device)
            genes_predict = autoencoder.predict(genes_control, emb_drugs).detach().cpu().numpy()
            mean_predict = genes_predict[:, :dim]
            var_predict = genes_predict[:, dim:]
            yp_m = mean_predict.mean(0)
            yp_v = var_predict.mean(0)
            y_true = np.array(dataset[idx].X.todense())
            yt_m = y_true.mean(axis=0)
            yt_v = y_true.var(axis=0)

            mean_score.append(r2_score(yt_m, yp_m))
            var_score.append(r2_score(yt_v, yp_v))
            mean_score_de.append(r2_score(yt_m[de_idx], yp_m[de_idx]))
            var_score_de.append(r2_score(yt_v[de_idx], yp_v[de_idx]))

            if for_plot:
                print([pert_categories[-1], mean_score[-1], mean_score_de[-1], var_score[-1], var_score_de[-1]])
                full_results.append([pert_categories[-1], mean_score[-1], mean_score_de[-1], var_score[-1], var_score_de[-1]])

    if for_plot:
        return full_results
    return [np.mean(s) if len(s) else -1 for s in [mean_score, mean_score_de, var_score, var_score_de]]


def evaluate_mse(autoencoder, dataset, genes_control, for_plot=False):
    """Evaluate MSE and Pearson correlation for means, including DE genes."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mean_score, var_score, mean_score_de, var_score_de = [], [], [], []
    num, dim = genes_control.size(0), genes_control.size(1)
    full_results = []
    pert_categories = []

    for pert_category in np.unique(dataset.obs.pert_categories):
        pert_categories.append(pert_category)
        idx = dataset.obs.pert_categories == pert_category
        de_idx = np.where(dataset.var_names.isin(np.array(dataset.uns["rank_genes_groups_cov"][str(pert_category)])))[0]

        if np.sum(idx) > 30:
            emb_drugs = torch.Tensor(
                dataset[idx].obs[["Drug1_numeric", "Drug2_numeric"]].to_numpy()[0:1].repeat(num, 0)
            ).to(device)
            genes_predict = autoencoder.predict(genes_control, emb_drugs).detach().cpu().numpy()
            mean_predict = genes_predict[:, :dim]
            var_predict = genes_predict[:, dim:]
            yp_m = mean_predict.mean(0)
            yp_v = var_predict.mean(0)
            y_true = np.array(dataset[idx].X.todense())
            yt_m = y_true.mean(axis=0)
            yt_v = y_true.var(axis=0)

            mean_score.append(mean_squared_error(yt_m, yp_m))
            var_score.append(pearsonr(yt_m, yp_m)[0])
            mean_score_de.append(mean_squared_error(yt_m[de_idx], yp_m[de_idx]))
            var_score_de.append(pearsonr(yt_m[de_idx], yp_m[de_idx])[0])

            if for_plot:
                print([pert_categories[-1], mean_score[-1], mean_score_de[-1], var_score[-1], var_score_de[-1]])
                full_results.append([pert_categories[-1], mean_score[-1], mean_score_de[-1], var_score[-1], var_score_de[-1]])

    if for_plot:
        return full_results
    return [np.mean(s) if len(s) else -1 for s in [mean_score, mean_score_de, var_score, var_score_de]]


def fit_CODEX_reconstruction_r2(args, dl_train_treated, dl_train_vehicle, vehicle_test, test_data, ood_data):
    log = []
    log.append(
        [
            "epoch",
            "train_loss",
            "R2_mean_val",
            "R2_mean_DEG_val",
            "R2_var_val",
            "R2_var_DEG_val",
            "R2_mean_ood",
            "R2_mean_DEG_ood",
            "R2_var_ood",
            "R2_var_DEG_ood",
        ]
    )
    print(log[-1])

    net = CODEXReconstruction(
        in_features=args["num_features"],
        num_nodes=args["layers"],
        num_treatments=args["num_treatments"],
        batch_norm=args["batch_norm"],
        dropout=args["dropout"],
        random_seed=args["seed"],
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net.to(device)

    optimizer_all = torch.optim.Adam(net.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
    optimizer_latent = torch.optim.Adam(net.T_rep.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])

    best_net = None
    best_val_accuracy = -np.inf
    early_stopping_count = 0

    for epoch in range(args["epochs"]):
        train_loss = 0.0
        for _, data in enumerate(dl_train_treated, 0):
            x = next(dl_train_vehicle.__iter__())[0]
            y, treatment = data[0], data[1]

            optimizer_all.zero_grad()
            gene_reconstructions = net(x, treatment)
            dim = gene_reconstructions.size(1) // 2
            gene_means = gene_reconstructions[:, :dim]
            gene_vars = gene_reconstructions[:, dim:]
            deg_weights = _batch_deg_weights(args, data, dim, gene_means.device)
            lossA = weighted_gaussian_nll_loss(gene_means, y, gene_vars, deg_weights)
            lossA.backward()
            optimizer_all.step()
            train_loss += lossA.item()

            if args["fine_tuning"]:
                optimizer_latent.zero_grad()
                gene_reconstructions = net(x, treatment)
                dim = gene_reconstructions.size(1) // 2
                gene_means = gene_reconstructions[:, :dim]
                gene_vars = gene_reconstructions[:, dim:]
                deg_weights = _batch_deg_weights(args, data, dim, gene_means.device)
                lossB = weighted_gaussian_nll_loss(gene_means, y, gene_vars, deg_weights)
                lossB.backward()
                optimizer_latent.step()
                train_loss += lossB.item()

        with torch.no_grad():
            r2_test = evaluate_r2_v2(net, test_data, vehicle_test)
            r2 = evaluate_r2_v2(net, ood_data, vehicle_test)

        log.append([epoch, train_loss, r2_test[0], r2_test[1], r2_test[2], r2_test[3], r2[0], r2[1], r2[2], r2[3]])
        print(log[-1])

        if r2_test[1] > best_val_accuracy:
            early_stopping_count = 0
            best_val_accuracy = r2_test[1]
            best_net = copy.deepcopy(net)

        if early_stopping_count > args["patience"]:
            break
        early_stopping_count += 1

    if not os.path.exists("{}/{}/".format(args["save_folder"], args["experiment_description"])):
        os.makedirs("{}/{}/".format(args["save_folder"], args["experiment_description"]))
    torch.save(best_net, "{}/{}/model.pt".format(args["save_folder"], args["experiment_description"]))
    pd.DataFrame(log).to_csv(
        "{}/{}/log.csv".format(args["save_folder"], args["experiment_description"]), index=False, header=False
    )
    return net


def fit_CODEX_reconstruction_mse(args, dl_train_treated, dl_train_vehicle, vehicle_test, test_data, ood_data=None):
    log = []
    log.append(
        [
            "epoch",
            "train_loss",
            "MSE_test",
            "MSE_DEG_test",
            "Pearson_test",
            "Pearson_DEG_test",
            "MSE_ood",
            "MSE_DEG_ood",
            "Pearson_ood",
            "Pearson_DEG_ood",
        ]
    )
    print(log[-1])

    net = CODEXReconstruction(
        in_features=args["num_features"],
        num_nodes=args["layers"],
        num_treatments=args["num_treatments"],
        batch_norm=args["batch_norm"],
        dropout=args["dropout"],
        random_seed=args["seed"],
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net.to(device)

    optimizer_all = torch.optim.Adam(net.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
    optimizer_latent = torch.optim.Adam(net.T_rep.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])

    best_net = None
    best_val_accuracy = np.inf
    early_stopping_count = 0

    for epoch in range(args["epochs"]):
        train_loss = 0.0
        for _, data in enumerate(dl_train_treated, 0):
            x = next(dl_train_vehicle.__iter__())[0]
            y, treatment = data[0], data[1]

            optimizer_all.zero_grad()
            gene_reconstructions = net(x, treatment)
            dim = gene_reconstructions.size(1) // 2
            gene_means = gene_reconstructions[:, :dim]
            gene_vars = gene_reconstructions[:, dim:]
            deg_weights = _batch_deg_weights(args, data, dim, gene_means.device)
            lossA = weighted_gaussian_nll_loss(gene_means, y, gene_vars, deg_weights)
            lossA.backward()
            optimizer_all.step()
            train_loss += lossA.item()

            if args["fine_tuning"]:
                optimizer_latent.zero_grad()
                gene_reconstructions = net(x, treatment)
                dim = gene_reconstructions.size(1) // 2
                gene_means = gene_reconstructions[:, :dim]
                gene_vars = gene_reconstructions[:, dim:]
                deg_weights = _batch_deg_weights(args, data, dim, gene_means.device)
                lossB = weighted_gaussian_nll_loss(gene_means, y, gene_vars, deg_weights)
                lossB.backward()
                optimizer_latent.step()
                train_loss += lossB.item()

        with torch.no_grad():
            r2_test = evaluate_mse(net, test_data, vehicle_test)
            if ood_data is not None:
                r2 = evaluate_mse(net, ood_data, vehicle_test)
            else:
                r2 = [-1, -1, -1, -1]

        log.append([epoch, train_loss, r2_test[0], r2_test[1], r2_test[2], r2_test[3], r2[0], r2[1], r2[2], r2[3]])
        print(log[-1])

        if r2_test[1] < best_val_accuracy:
            early_stopping_count = 0
            best_val_accuracy = r2_test[1]
            best_net = copy.deepcopy(net)

        if early_stopping_count > args["patience"]:
            break
        early_stopping_count += 1

    if not os.path.exists("{}/{}/".format(args["save_folder"], args["experiment_description"])):
        os.makedirs("{}/{}/".format(args["save_folder"], args["experiment_description"]))
    torch.save(best_net, "{}/{}/model.pt".format(args["save_folder"], args["experiment_description"]))
    pd.DataFrame(log).to_csv(
        "{}/{}/log.csv".format(args["save_folder"], args["experiment_description"]), index=False, header=False
    )
    return net
