import sys
sys.path.append("/home/vmeijer/contrail-height-estimation/src/")

from training import *
from models import * 
from dataset import *

import pandas as pd, numpy as np
import json
import wandb
import argparse


def run_experiment(config, device):

    use_wandb = True
    gpu = True

    df = pd.read_pickle(config["dataset"])

    print(f"Retrieved dataset with {len(df)} entries.")

    target_col = config["target_col"]
    cols = config["input_cols"]
    
    # Always use same seed for numpy such that dataset splits is the same across
    # an ensemble
    np.random.seed(0) #config["seed"])
    torch.manual_seed(config['seed'])

    n = len(df)

    if config["overfit"]:
        train = np.random.choice(range(n), size=int(0.8*n), replace=False)
        val = np.arange(n)[~np.isin(np.arange(n), train)]
    

        target_col = config["target_col"]
        cols = config["input_cols"]

        df["goes_time"] = pd.to_datetime(df["goes_time"])


        ds_train = TabularDataset(df[cols + [target_col]].iloc[:100], [], cols, [target_col])
        ds_val = TabularDataset(df[cols + [target_col]].iloc[:100], [], cols, [target_col], mapper=ds_train.cont_mapper)
        
    else:
        train = np.random.choice(range(n), size=int(0.8*n), replace=False)
        val = np.arange(n)[~np.isin(np.arange(n), train)]
    

        target_col = config["target_col"]
        cols = config["input_cols"]

        df["goes_time"] = pd.to_datetime(df["goes_time"])


        ds_train = TabularDataset(df[cols + [target_col]].iloc[train], [], cols, [target_col], scale_target=config["scale_target"])
        ds_val = TabularDataset(df[cols + [target_col]].iloc[val], [], cols, [target_col], mapper=ds_train.cont_mapper, scale_target=config["scale_target"])

    if use_wandb:
        wandb.init(project="height-estimation", config=config)

    n_inputs = len(config["input_cols"])

    # Compute hidden layer sizes
    hidden_layer_sizes = [config["first_layer_size"]]
    for i in range(config["n_layers"]):
        hidden_layer_sizes.append(max(1, int(hidden_layer_sizes[-1]*config["layer_size_factor"])))

    model = RegressionMLP(n_inputs, 1, hidden_layer_sizes, activation=config['activation'],
                            dropout_p=config["dropout_p"])
    print(model)
    if gpu:
        model.to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                    weight_decay=config['weight_decay'])
        
    use_scheduler = config["use_scheduler"]
    if use_scheduler: 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config['patience'],
                                                            factor=config['factor'])
    else:
        scheduler = None

    if len(config["checkpoint"]) > 0:
        checkpoint = torch.load(config["checkpoint"])
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        for param_group in optimizer.param_groups:
            param_group["lr"] = config["lr"]

    trained_model = train_deterministic_model(model, {"train":ds_train, "val": ds_val}, nn.MSELoss(),
                                optimizer, config, device,
                                scheduler=scheduler, use_wandb=use_wandb)

    # Save config
    with open ("/net/d13/data/vmeijer/contrail-height-estimation/saved_configs/" + wandb.run.name + ".json", "w+") as f:
        json.dump(config, f)
    # Save model
    torch.save(trained_model, '/net/d13/data/vmeijer/contrail-height-estimation/saved_models/' + wandb.run.name + '.pt')

def main():

    

    experiment_config = json.load(open(sys.argv[1], 'r'))
    if len(sys.argv) > 2:
        device = torch.device(sys.argv[-1])
    else:
        device = torch.device("cuda")
    run_experiment(experiment_config, device)


if __name__ == "__main__":
    main() 