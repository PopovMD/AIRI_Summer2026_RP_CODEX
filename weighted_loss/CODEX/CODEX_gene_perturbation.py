import argparse

import torch

from codex.CODEX_reconstruction import *
from codex.reconstruction_utils import *


device = "cuda:0" if torch.cuda.is_available() else "cpu"

"""
Example runs from the original CODEX repository:

docker run -it --rm --gpus "device=0" -v /sybig/home/ssc/CODEX:/mnt codex python3 -i CODEX_gene_perturbation.py -l 512 128 64 -s 1 -dn norman

docker run -it --rm --gpus "device=0" -v /sybig/home/ssc/CODEX:/mnt codex python3 -i CODEX_gene_perturbation.py -l 512 128 64 -s 1 -dn replogle_rpe1_essential --download_data
docker run -it --rm --gpus "device=0" -v /sybig/home/ssc/CODEX:/mnt codex python3 -i CODEX_gene_perturbation.py -l 512 128 64 -s 1 -dn replogle_k562_essential --download_data

DEG-weighted Gaussian NLL run, enabled by default in this modified file:

docker run -it --rm --gpus "device=0" -v /sybig/home/ssc/CODEX:/mnt codex python3 -i CODEX_gene_perturbation.py -l 512 128 64 -s 1 -dn norman --deg_weighting
"""


def parse_arguments():
    parser = argparse.ArgumentParser(description="Gene perturbation experiments.")

    # Training arguments
    parser.add_argument("--num_features", type=int, default=5045)
    parser.add_argument("--num_treatments", type=int, default=None)
    parser.add_argument("-ft", "--fine_tuning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-l", "--layers", nargs="+", type=int, required=True)  # [512, 128, 64]
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-bs", "--batch_size", type=int, default=256)
    parser.add_argument("-do", "--dropout", type=float, default=0.1)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.000001)
    parser.add_argument("-bn", "--batch_norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-e", "--epochs", type=int, default=1000)
    parser.add_argument("-p", "--patience", type=int, default=50)
    parser.add_argument("-s", "--seed", type=int, default=1)
    parser.add_argument("-exp", "--experiment", type=str, default="simulation")
    parser.add_argument("-dp", "--data_path", type=str, default="/mnt/data/")
    parser.add_argument(
        "-dn",
        "--data_name",
        type=str,
        default="norman",
        help='GEARS dataset name, e.g. "norman", "replogle_rpe1_essential", "replogle_k562_essential".',
    )
    parser.add_argument("--download_data", action=argparse.BooleanOptionalAction, default=False)

    # DEG weighting arguments.
    # For each perturbation condition c and each DEG g from adata.uns['rank_genes_groups_cov'][c]:
    #   w(c, g) = 1 + min(deg_weight_max, deg_weight_alpha * |mean_c(g)-mean_ctrl(g)| / deg_effect_scale)
    #                 * clip((deg_top_k - rank(g)) / deg_top_k, 0, 1)
    # Non-DEG genes keep weight 1.0. The weighted NLL is normalized by sum(weights).
    parser.add_argument(
        "--deg_weighting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable per-gene DEG weights in the Gaussian NLL loss.",
    )
    parser.add_argument(
        "--deg_weight_alpha",
        type=float,
        default=1.0,
        help="Multiplier applied to the absolute condition-vs-control mean shift in DEG weights.",
    )
    parser.add_argument(
        "--deg_weight_max",
        type=float,
        default=5.0,
        help="Maximum extra multiplicative weight above 1.0 for a DEG before rank clipping.",
    )
    parser.add_argument(
        "--deg_effect_scale",
        type=float,
        default=1.0,
        help="Scale for |mean_condition - mean_control| in DEG weights. Increase to make weights less aggressive.",
    )
    parser.add_argument(
        "--deg_top_k",
        type=int,
        default=20,
        help="Number of ranked DEGs per condition to use from adata.uns['rank_genes_groups_cov'].",
    )

    # Experiment folder
    parser.add_argument("--save_folder", type=str, default="/mnt/models/")

    return dict(vars(parser.parse_args()))


if __name__ == "__main__":
    args = parse_arguments()

    args["save_folder"] = (
        args["save_folder"]
        + args["data_name"]
        + "_"
        + args["experiment"]
        + "_seed"
        + str(args["seed"])
    )
    args["experiment_description"] = (
        "CODEX,l={},ft={},lr={},bs={},do={},wd={},bn={},e={},p={},s={},degw={},dega={},degmax={},degscale={},degtopk={}"
    ).format(
        args["layers"],
        args["fine_tuning"],
        args["learning_rate"],
        args["batch_size"],
        args["dropout"],
        args["weight_decay"],
        args["batch_norm"],
        args["epochs"],
        args["patience"],
        args["seed"],
        args["deg_weighting"],
        args["deg_weight_alpha"],
        args["deg_weight_max"],
        args["deg_effect_scale"],
        args["deg_top_k"],
    )

    print(args["experiment_description"])

    train_dl, control_train_dl, control_test, test_data, adata, num_treatments, _, _ = load_data(
        args, data_name=args["data_name"]
    )
    # NOTE: adata.obs.subgroup defines whether the samples are used for training or one of the testing scenarios
    # (['combo_seen2', 'combo_seen1', 'combo_seen0', 'unseen_single'])

    args["num_treatments"] = num_treatments
    args["num_features"] = test_data.shape[1]

    print(num_treatments)

    net = fit_CODEX_reconstruction_mse(args, train_dl, control_train_dl, control_test, test_data)
