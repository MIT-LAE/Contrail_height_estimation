import torch
import torch.nn as nn 
import torch.nn.functional as F
import wandb 
import matplotlib.pyplot as plt

from quantnn.models.pytorch.common import QuantileLoss
from quantnn.quantiles import posterior_mean

import pytorch_warmup as warmup
from contrails.satellites.goes.abi import get_ash

from dataset import *

channel_means = np.array([277.8691272 , 231.85395151, 239.58358231, 246.85718799,
       265.97517924, 249.26745457, 267.63629401, 266.38283615,
       264.02629492, 253.99376392])
channel_std_devs = np.array([20.32268558,  7.60601058, 10.17442007, 12.85312767, 22.31384306,
       15.39341491, 23.32276947, 23.5053471 , 22.75305663, 17.69552913])

def freeze_batchnorm_statistics(model):

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            # if hasattr(module, 'weight'):
            #     module.weight.requires_grad_(False)
            # if hasattr(module, 'bias'):
            #     module.bias.requires_grad_(False)
            module.eval()


def visualize_predictions(inputs, labels, predictions, contrail=False):

    if contrail:
        vmin = 8
        vmax = 15
    else:
        vmin = 6
        vmax = 16

    fig, axes = plt.subplots(dpi=300, ncols=5, nrows=5, figsize=(12.5, 12.5))

    for i in range(5):
        # Get ash image
        ch11 = channel_std_devs[4] * inputs[i,4,:,:] + channel_means[4]
        ch14 = channel_std_devs[7] * inputs[i,7,:,:] + channel_means[7]
        ch15 = channel_std_devs[8] * inputs[i,8,:,:] + channel_means[8]

        ash = get_ash(ch11, ch14, ch15)

        axes[i,0].imshow(ash)
        # Get location of collocations (use 1 as threshold for safety since numerical
        # errors might have caused the 0 values to be slightly larger than 0)
        rows, cols = np.where(labels[i,0,:,:] > 1)
        axes[i,0].scatter(cols, rows, s=1, c="r", marker="x")
    
        axes[i,1].imshow(labels[i,0,:,:], vmin=vmin, vmax=vmax)
        axes[i,2].imshow(predictions[i,0,:,:], vmin=vmin, vmax=vmax)
        
        axes[i,3].imshow(inputs[i,-1,:,:], cmap="gray")


        axes[i,4].scatter(labels[i,0, rows, cols], predictions[i,0,rows,cols], alpha=0.5, c="b", s=1)
        axes[i,4].plot([vmin, vmax], [vmin, vmax], c="k", linestyle="dashed")
        axes[i,4].set_xlim(vmin, vmax)
        axes[i,4].set_ylim(vmin, vmax)

        axes[i,4].set(ylabel="Prediction, km")
        axes[i,4].set(xlabel="Truth, km")
        axes[i,4].set_aspect("equal")


        for j in range(4):
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])

    plt.tight_layout()
   
    return fig


def get_datasets(config, return_indices=True):
    """
    Returns the training and validation dataset for the specified 'config'.

    Parameters
    ----------
    config: dict
        Training configuration
    return_indices: bool (optional)
        Whether or not to return the dataframe indices used

    Returns
    -------
    ds_train: torch.utils.data.Dataset
        Training dataset object
    ds_val: torch.utils.data.Dataset
        Validation dataset object
    """

    df = pd.read_pickle(config["dataset"])
    df["goes_time"] = pd.to_datetime(df["goes_time"])
    df = df.drop(columns=["L1_file", "goes_product", "adv_time", "caliop_time", "manual_inspection"])

    df = df.groupby(["goes_time", "goes_ABI_row", "goes_ABI_col"]).mean().reset_index()

    target_col = config["target_col"]
    cols = config["input_cols"]

    # Perform seed
    np.random.seed(config["numpy_seed"])
    torch.manual_seed(config['seed'])

    unique_times = df.goes_time.unique()


    n = len(unique_times)
    train = np.random.choice(range(n), size=int(0.8*n), replace=False)
    
    not_in_train = np.arange(n)[~np.isin(np.arange(n), train)]
    val = np.random.choice(not_in_train, size=(int(0.5*len(not_in_train))), replace=False)


    df["goes_time"] = pd.to_datetime(df["goes_time"])


    train_indices = df.goes_time.isin(unique_times[train])
    val_indices = df.goes_time.isin(unique_times[val])
   
    ds_train = TabularDataset(df[cols + [target_col]][train_indices],
                                [], cols, [target_col], scale_target=config["scale_target"])
    ds_val = TabularDataset(df[cols + [target_col]][val_indices], [], cols, [target_col],
    mapper=ds_train.cont_mapper, scale_target=config["scale_target"])

    if return_indices:
        return ds_train, ds_val, train_indices, val_indices
    else:
        return ds_train, ds_val
    

def train_deterministic_model(model, datasets, loss, optimizer, config, device, scheduler=None, use_wandb=True, gpu=True):
    
    batch_size = config["batch_size"]

    
    # Provide information on scaled losses as well since these directly relate
    # to the RMSE / MSE
    if config["scale_target"]:
        
            
        target_var = config["target_std_dev"]**2

        # If we're using L1 loss, we should scale by the std dev
        if config["loss"] == "l1":
            target_var = target_var ** 0.5
        
    else:
        target_var = 1

    if config["warmup"]:
        warmup_scheduler = warmup.LinearWarmup(optimizer,
                                                warmup_period=config["warmup_length"])


    if config["weighted_sampling"]:
        if config["cirrus"]:
            sample_weights = np.load("/home/vmeijer/contrail-height-estimation/data/cirrus_mlp_weights.npy")
        else:
            sample_weights = np.load("/home/vmeijer/contrail-height-estimation/data/mlp_train_sample_weights.npy")

        sampler = torch.utils.data.WeightedRandomSampler(sample_weights.flatten(),
                                                        len(sample_weights))
        shuffle = False
    else:
        sampler = None
        shuffle = True

    dataloader_train = torch.utils.data.DataLoader(datasets["train"],
                            batch_size=batch_size, num_workers=config["num_workers"], shuffle=shuffle, sampler=sampler,
                            pin_memory=True)
    dataloader_val = torch.utils.data.DataLoader(datasets["val"],
                                            batch_size=batch_size, num_workers=config["num_workers"],
                                            pin_memory=True)
    
    if use_wandb:
        wandb.watch(model, log="all", log_freq=10)

    for epoch in range(config['n_epochs']):
        model.train()

        if epoch >= config["freeze_bn_after"]:
            freeze_batchnorm_statistics(model)

        counter = 0
        train_loss = 0 
        for x, y in dataloader_train:
            if gpu:
                x = x.to(device)
                y = y.to(device)

            optimizer.zero_grad()
            mean = model(x.float())
           
            loss_val = torch.mean(loss(mean, y.float()))
            
            loss_val.backward()
            optimizer.step()
            
            counter += x.shape[0]
            train_loss += x.shape[0]*loss_val.item()

        train_loss /= counter
        
        model.eval()
        counter = 0
        val_loss = 0
        with torch.no_grad():
            for x, y in dataloader_val:
                
                if gpu:
                    x = x.to(device)
                    y = y.to(device)
                mean = model(x.float())

                loss_val = torch.mean(loss(mean, y.float()))
                counter += x.shape[0]
                val_loss += x.shape[0]*loss_val.item()
          
        if use_wandb:
            wandb.log({"epoch" : epoch, "scaled_train_loss" : target_var * train_loss, "train_loss" : train_loss, "val_loss" : val_loss/counter,
                        "scaled_val_loss" : target_var * val_loss/counter, "lr" : optimizer.param_groups[0]["lr"]
                        })
            
        if config["warmup"]:
            with warmup_scheduler.dampening():
                if scheduler is not None:
                    scheduler.step(val_loss)
        else:
            if scheduler is not None:
                scheduler.step(val_loss)

        print(f"Epoch {epoch}, Training loss {train_loss}, Validation loss {val_loss/counter}, Scaled training loss {train_loss*target_var}, Scaled val loss {val_loss * target_var / counter}")

        if epoch % config["checkpoint_interval"] == 0 and epoch > 0:
            checkpoint = {'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()}
            torch.save(checkpoint, "/net/d15/data/vmeijer/contrail-height-estimation/checkpoints/" + wandb.run.name + f"_epoch_{epoch}.pt")

    return model

def train_quantile_convolutional_model(model, datasets, optimizer, config, device, quantiles, scheduler=None, use_wandb=True, gpu=True):
    
    batch_size = config["batch_size"]

    if "contrail" in config["dataset"]:
        contrail = True
    else:
        contrail = False

    mse_loss = nn.MSELoss()

    if config["pretrained_cirrus"]:
        target_mean = config["target_mean"]
        target_std_dev = config["target_std_dev"]
        # target_mean = config["cirrus_target_mean"]
        # target_std_dev = config["cirrus_target_std_dev"]
    else:
        target_mean = config["target_mean"]
        target_std_dev = config["target_std_dev"]


    if config["warmup"]:
        warmup_scheduler = warmup.LinearWarmup(optimizer,
                                                warmup_period=config["warmup_length"])

    torch.manual_seed(config["seed"])
    
    if config["weighted_sampling"]:

        if config['gfs']:
            sample_weights = np.load("/home/vmeijer/contrail-height-estimation/data/train_sample_weights_gfs.npy")
        else:
            sample_weights = np.load("/home/vmeijer/contrail-height-estimation/data/train_sample_weights.npy")

        sampler = torch.utils.data.WeightedRandomSampler(sample_weights.flatten(),
                                                        len(sample_weights))
        shuffle = False
    else:
        sampler = None
        shuffle = True

    dataloader_train = torch.utils.data.DataLoader(datasets["train"],
                            batch_size=batch_size, num_workers=config["num_workers"], shuffle=shuffle,
                            pin_memory=True, sampler=sampler)
    
    dataloader_val = torch.utils.data.DataLoader(datasets["val"],
                                            batch_size=batch_size, num_workers=config["num_workers"],
                                            pin_memory=True)
    
    if use_wandb:
        wandb.watch(model, log="all", log_freq=10)

    loss = QuantileLoss(quantiles, mask=-9999)


    for epoch in range(config['n_epochs']):
        model.train()
        counter = 0
        train_loss = 0
        mse_train_loss = 0
        visualized = False
        for x, y in dataloader_train:


            if epoch >= config["freeze_bn_after"]:
                freeze_batchnorm_statistics(model)
            

            mask = y <= 0

            y = (y-target_mean) / target_std_dev

            y[mask] = -9999


            if gpu:
                x = x.to(device)
                y = y.to(device)
                mask = mask.to(device)
            optimizer.zero_grad()

        
            
            pred_quantiles = model(x.float())
            mean = posterior_mean(pred_quantiles, torch.Tensor(quantiles).to(pred_quantiles.device)).unsqueeze(1)
            if not visualized:
                fig = visualize_predictions(x.detach().cpu().numpy(), target_std_dev * y.detach().cpu().numpy() + target_mean,
                                                 target_std_dev * mean.detach().cpu().numpy() + target_mean, contrail=contrail)
                wandb.log({"train_viz": fig})
                visualized = True

    

            loss_val = loss(pred_quantiles, y.float())
            mse_loss_val = torch.mean(mse_loss(mean[y != -9999], y.float()[y != -9999]))
            loss_val.backward()
            optimizer.step()
            
            counter += x.shape[0]
            train_loss += x.shape[0]*loss_val.item()
            mse_train_loss += x.shape[0] * mse_loss_val.item()

        train_loss /= counter
        mse_train_loss /= counter
        
        model.eval()
        counter = 0
        val_loss = 0
        mse_val_loss = 0
        visualized = False
        with torch.no_grad():
            for x, y in dataloader_val:
                mask = y <= 0

                y = (y-target_mean) / target_std_dev

                y[mask] = -9999
                    
                if gpu:
                    x = x.to(device)
                    y = y.to(device)
                    mask = mask.to(device)

                pred_quantiles = model(x.float())
                mean = posterior_mean(pred_quantiles, torch.Tensor(quantiles).to(pred_quantiles.device)).unsqueeze(1)

                if not visualized:
                    fig = visualize_predictions(x.detach().cpu().numpy(), target_std_dev * y.detach().cpu().numpy() + target_mean,
                                                target_std_dev * mean.detach().cpu().numpy() + target_mean, contrail=contrail)
                    wandb.log({"val_viz": fig})
                    visualized = True
    
                loss_val = loss(pred_quantiles, y.float())
                mse_loss_val = torch.mean(mse_loss(mean[y != -9999], y.float()[y != -9999]))
                counter += x.shape[0]
                val_loss += x.shape[0]*loss_val.item()
                mse_val_loss += x.shape[0] * mse_loss_val.item()
          
        if use_wandb:
            wandb.log({"epoch" : epoch, "mse_train_loss" : target_std_dev**2 * mse_train_loss, "train_loss" : train_loss, "val_loss" : val_loss/counter,
                        "mse_val_loss" : target_std_dev**2 * mse_val_loss/counter, "lr" : optimizer.param_groups[0]["lr"]
                        })
            
            
        if config["warmup"]:
            with warmup_scheduler.dampening():
                if scheduler is not None:
                    scheduler.step(val_loss)
        else:
            if scheduler is not None:
                scheduler.step(val_loss)

        print(f"Epoch {epoch}, Training loss {train_loss}, Validation loss {val_loss/counter}")

        if epoch % config["checkpoint_interval"] == 0 and epoch > 0:
            checkpoint = {'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()}
            torch.save(checkpoint, "/net/d15/data/vmeijer/contrail-height-estimation/checkpoints/" + wandb.run.name + f"_epoch_{epoch}.pt")


    return model



def train_convolutional_model(model, datasets, loss, optimizer, config, device, scheduler=None, use_wandb=True, gpu=True):
    
    batch_size = config["batch_size"]

    if "contrail" in config["dataset"]:
        contrail = True
    else:
        contrail = False


    if config["pretrained_cirrus"]:
        target_mean = config["target_mean"]
        target_std_dev = config["target_std_dev"]
        # target_mean = config["cirrus_target_mean"]
        # target_std_dev = config["cirrus_target_std_dev"]
    else:
        target_mean = config["target_mean"]
        target_std_dev = config["target_std_dev"]


    if config["warmup"]:
        warmup_scheduler = warmup.LinearWarmup(optimizer,
                                                warmup_period=config["warmup_length"])

    torch.manual_seed(config["seed"])
    
    if config["weighted_sampling"]:

        if config['gfs']:
            sample_weights = np.load("/home/vmeijer/contrail-height-estimation/data/train_sample_weights_gfs.npy")
        else:

            if config["cirrus"]:
                sample_weights = np.load("/home/vmeijer/contrail-height-estimation/data/cirrus_conv_weights_v2.npy")
            else:
                sample_weights = np.load("/home/vmeijer/contrail-height-estimation/data/train_sample_weights.npy")

        sampler = torch.utils.data.WeightedRandomSampler(sample_weights.flatten(),
                                                        len(sample_weights))
        shuffle = False
    else:
        sampler = None
        shuffle = True

    dataloader_train = torch.utils.data.DataLoader(datasets["train"],
                            batch_size=batch_size, num_workers=config["num_workers"], shuffle=shuffle,
                            pin_memory=True, sampler=sampler)
    
    dataloader_val = torch.utils.data.DataLoader(datasets["val"],
                                            batch_size=batch_size, num_workers=config["num_workers"],
                                            pin_memory=True)
    
    if use_wandb:
        wandb.watch(model, log="all", log_freq=10)


    if config["pretrained_cirrus"]:
        model.eval()
        counter = 0
        val_loss = 0

        with torch.no_grad():
            for x, y in dataloader_val:
                mask = y > 0
                y = (y - target_mean) / target_std_dev
                
                if gpu:
                    x = x.to(device)
                    y = y.to(device)
                    mask = mask.to(device)
                mean = model(x.float())

                loss_val = torch.mean(loss(mean[mask], y.float()[mask]))
            
                counter += x.shape[0]
                val_loss += x.shape[0]*loss_val.item()

        print(f"Before training, validation loss is {target_std_dev**2 * val_loss/counter}")
        
        

    for epoch in range(config['n_epochs']):
        model.train()
        counter = 0
        train_loss = 0
        visualized = False
        for x, y in dataloader_train:


            if epoch >= config["freeze_bn_after"]:
                freeze_batchnorm_statistics(model)
            

            mask = y > 0

            y = (y-target_mean) / target_std_dev

            if gpu:
                x = x.to(device)
                y = y.to(device)
                mask = mask.to(device)
            optimizer.zero_grad()

            if config["upscale"]:
                x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            mean = model(x.float())

            if config['upscale']:
                mean = F.interpolate(mean, scale_factor = 1/2, mode="bilinear", align_corners=False)

            if not visualized:
                fig = visualize_predictions(x.detach().cpu().numpy(), target_std_dev * y.detach().cpu().numpy() + target_mean,
                                                 target_std_dev * mean.detach().cpu().numpy() + target_mean, contrail=contrail)
                wandb.log({"train_viz": fig})
                visualized = True


            loss_val = torch.mean(loss(mean[mask], y.float()[mask]))

            loss_val.backward()
            optimizer.step()
            
            counter += x.shape[0]
            train_loss += x.shape[0]*loss_val.item()

        train_loss /= counter
        
        model.eval()
        counter = 0
        val_loss = 0
        visualized = False
        with torch.no_grad():
            for x, y in dataloader_val:
                mask = y > 0
                y = (y - target_mean) / target_std_dev
                
                if gpu:
                    x = x.to(device)
                    y = y.to(device)
                    mask = mask.to(device)
                mean = model(x.float())
                if not visualized:
                    fig = visualize_predictions(x.detach().cpu().numpy(), target_std_dev * y.detach().cpu().numpy() + target_mean,
                                                target_std_dev * mean.detach().cpu().numpy() + target_mean, contrail=contrail)
                    wandb.log({"val_viz": fig})
                    visualized = True


                loss_val = torch.mean(loss(mean[mask], y.float()[mask]))
            
                counter += x.shape[0]
                val_loss += x.shape[0]*loss_val.item()
          
        if use_wandb:
            wandb.log({"epoch" : epoch, "scaled_train_loss" : target_std_dev**2 * train_loss, "train_loss" : train_loss, "val_loss" : val_loss/counter,
                        "scaled_val_loss" : target_std_dev**2 * val_loss/counter, "lr" : optimizer.param_groups[0]["lr"]
                        })
            
            
        if config["warmup"]:
            with warmup_scheduler.dampening():
                if scheduler is not None:
                    scheduler.step(val_loss)
        else:
            if scheduler is not None:
                scheduler.step(val_loss)

        print(f"Epoch {epoch}, Training loss {train_loss}, Validation loss {val_loss/counter}")

        if epoch % config["checkpoint_interval"] == 0 and epoch > 0:
            checkpoint = {'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()}
            torch.save(checkpoint, "/net/d15/data/vmeijer/contrail-height-estimation/checkpoints/" + wandb.run.name + f"_epoch_{epoch}.pt")


    return model
