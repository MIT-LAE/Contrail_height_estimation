import torch
import torch.nn as nn 
import wandb 
import matplotlib.pyplot as plt

def train_deterministic_model(model, datasets, loss, optimizer, config, device, scheduler=None, use_wandb=True, gpu=True):
    
    batch_size = config["batch_size"]

    # Provide information on scaled losses as well since these directly relate
    # to the RMSE / MSE
    if config["scale_target"]:
        target_var = datasets["train"].cont_mapper.features[-1][1].var_[0]
    else:
        target_var = 1

    dataloader_train = torch.utils.data.DataLoader(datasets["train"],
                            batch_size=batch_size, num_workers=4, shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(datasets["val"],
                                            batch_size=batch_size, num_workers=4)
    
    if use_wandb:
        wandb.watch(model)

    for epoch in range(config['n_epochs']):
        model.train()
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
        if scheduler is not None:
            scheduler.step(val_loss)

        print(f"Epoch {epoch}, Training loss {train_loss}, Validation loss {val_loss/counter}, Scaled training loss {train_loss*target_var}, Scaled val loss {val_loss * target_var / counter}")

        if epoch % config["checkpoint_interval"] == 0 and epoch > 0:
            checkpoint = {'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()}
            torch.save(checkpoint, "/net/d13/data/vmeijer/contrail-height-estimation/checkpoints/" + wandb.run.name + f"_epoch_{epoch}.pt")


    return model
