import torch
import torch.nn as nn
from AGCRN import AGCRN as Network
from BasicTrainer import Trainer
from lib.dataloader import get_dataloader
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import wandb
import os

import random
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import math

logging.basicConfig(level=logging.INFO)


def objective_function(tuned_args: DictConfig, args: DictConfig) -> float:
    # Random seed
    random.seed(args.seed)
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Device
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.device[5]))
    else:
        args.device = "cpu"

    print(args.device)

    # Logging
    logging.info(f"Embed Dimension: {args.embed_dim}")
    logging.info(f"Number of Layers: {args.num_layers}")
    logging.info(f"RNN Units: {args.rnn_units}")
    logging.info(f"Chebyshev Polynomial Order: {args.cheb_k}")
    logging.info(f"Initial Learning Rate: {args.lr_init}")
    # logging.info(f"Number of Heads: {args.num_heads}")
    logging.info(f"Hidden Dimension of Node: {args.hidden_dim_node}")
    logging.info(f"Number of Layers of Node: {args.num_layers_node}")
    # logging.info(f"Input Dimension: {args.input_dim}")

    # Number of parameters
    model = Network(args)
    model = model.to(args.device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    logging.info("Model parameters:")
    for name, param in model.named_parameters():
        logging.info(f"{name}: {param.shape}")
    logging.info(
        f"Total number of parameters: {sum(p.numel() for p in model.parameters())}"
    )

    # load dataset
    (
        train_loader,
        val_loader,
        test_loader,
        scaler,
    ) = get_dataloader(
        args,
        normalizer=args.normalizer,
        single=True,
    )

    # init loss function, optimizer
    loss = torch.nn.MSELoss().to(args.device)
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.lr_init,
        eps=1.0e-8,
        weight_decay=0,
        amsgrad=False,
    )

    # learning rate decay
    lr_scheduler = None
    if args.lr_decay:
        print("Applying learning rate decay.")
        lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(","))]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=lr_decay_steps, gamma=args.lr_decay_rate
        )

    if args.saved_model_path is not None:
        logging.info("Loading saved model from {}".format(args.saved_model_path))
        model.load_state_dict(torch.load(args.saved_model_path))
        model.to(args.device)
        # testing the model
        logging.info("Testing the model.")

        trainer = Trainer(
            model,
            loss,
            optimizer,
            train_loader,
            val_loader,
            test_loader,
            scaler,
            args,
            lr_scheduler=lr_scheduler,
        )

        test_loss = trainer.test(
            model, trainer.args, trainer.test_loader, trainer.scaler, trainer.logger
        )
        return {"loss": test_loss, "status": STATUS_OK}
    else:
        # start training
        trainer = Trainer(
            model,
            loss,
            optimizer,
            train_loader,
            val_loader,
            test_loader,
            scaler,
            args,
            lr_scheduler=lr_scheduler,
        )
        best_loss = trainer.train()

        return {"loss": best_loss, "status": STATUS_OK}


@hydra.main(
    version_base=None,
    config_path="../../configuration/modules",
    config_name="AGCRN_time_dependent_lapalcian",
)
def main(args: DictConfig) -> None:
    # Wandb
    os.environ["WANDB_DIR"] = args.wandb_dir
    if args.saved_model_path is not None:
        os.environ["WANDB_MODE"] = "dryrun"
    else:
        os.environ["WANDB_MODE"] = args.wandb_mode

    if not args.hyperopt_tuning:
        tuned_args = args
        each_run_name = args.run_name
        wandb.init(project="AGCRN_dynamic_tune", name=each_run_name)
        wandb.config = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)
        objective_function(tuned_args, args)
        wandb.run.finish()

    else:
        space = {
            "lr_init": hp.loguniform(
                "lr_init",
                math.log(1.0e-05),
                math.log(1.0e-02),
            ),
            "num_layers": hp.choice("num_layers", [1, 2]),
            # "cheb_k": hp.choice("cheb_k", [1, 2, 3]),
            "rnn_units": hp.choice("rnn_units", [8, 16, 32, 64]),
            "embed_dim": hp.choice("embed_dim", [8, 16, 32, 64]),
            "hidden_dim_node": hp.choice("hidden_dim_node", [2, 4, 8, 16, 32, 64, 128]),
            "num_layers_node": hp.choice("num_layers_node", [1, 2, 3, 4, 5, 6]),
            # "num_heads": hp.choice("num_heads", [1, 2, 4]),
            # "input_dim": hp.choice("input_dim", [2, 3, 4, 5, 6]),
        }

        trials = Trials()
        trial_counter = [0]

        def objective_with_args_fixed(tuned_args):
            # Starting a new wandb run
            each_run_name = args.run_name + f"_trial_{trial_counter[0]}"
            wandb.init(
                project="AGCRN_dynamic_tune", name=each_run_name, reinit=True
            )  # reinit=True allows multiple wandb.init() calls
            wandb.config = OmegaConf.to_container(
                args, resolve=True, throw_on_missing=True
            )
            result = objective_function(tuned_args, args)
            wandb.run.finish()
            trial_counter[0] += 1  # Increment the trial counter
            return result

        best = fmin(
            objective_with_args_fixed,
            space=space,
            algo=tpe.suggest,
            max_evals=args.n_trials,
            trials=trials,
        )

        logging.info(best)


if __name__ == "__main__":
    main()
