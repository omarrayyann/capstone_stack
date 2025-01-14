import random
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from vqvae.vqvae import *
from torch.nn import DataParallel

import wandb


def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)


@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    multi_gpu = False

    dataset = hydra.utils.instantiate(cfg.dataset)
    train_dataset, val_dataset = hydra.utils.instantiate(cfg.split_dataset, dataset=dataset)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    if "visual_input" in cfg and cfg.visual_input:
        print("use visual environment")
        cfg.model.gpt_model.config.input_dim = 1024
    cbet_model = hydra.utils.instantiate(cfg.model).to(cfg.device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        cbet_model = DataParallel(cbet_model)
        multi_gpu = True
    else:
        print("Using a single GPU.")
    
    if cfg.load_path:
        if multi_gpu:
            cbet_model.module.load_model(Path(cfg.load_path))
        else:
            cbet_model.load_model(Path(cfg.load_path))
    
    if multi_gpu:
        optimizer = cbet_model.module.configure_optimizers(
            weight_decay=cfg.optim.weight_decay,
            learning_rate=cfg.optim.lr,
            betas=cfg.optim.betas,
        )
    else:
        optimizer = cbet_model.configure_optimizers(
            weight_decay=cfg.optim.weight_decay,
            learning_rate=cfg.optim.lr,
            betas=cfg.optim.betas,
        )
        
        
    run = wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    save_path = Path(cfg.save_path)


    for epoch in tqdm.trange(cfg.epochs):
        cbet_model.eval()
        if epoch % cfg.eval_freq == 0:
            total_loss = 0
            action_diff = 0
            action_diff_tot = 0
            action_diff_mean_res1 = 0
            action_diff_mean_res2 = 0
            action_diff_max = 0
            with torch.no_grad():
                for data in val_loader:
                    act = data["action"].to(cfg.device)[:,-1,:].reshape(-1, 1, 8)
                    print(act.shape)

                    rgb = data["rgb"].to(cfg.device)
                    depth = data["depth"].to(cfg.device)

                    goal = None
                    if multi_gpu:
                        predicted_act, loss, loss_dict = cbet_model.module(rgb, depth, goal, act)
                    else:
                        predicted_act, loss, loss_dict = cbet_model(rgb, depth, goal, act)
                    total_loss += loss.item()
                    wandb.log({"eval/{}".format(x): y for (x, y) in loss_dict.items()})
                    action_diff += loss_dict["action_diff"]
                    action_diff_tot += loss_dict["action_diff_tot"]
                    action_diff_mean_res1 += loss_dict["action_diff_mean_res1"]
                    action_diff_mean_res2 += loss_dict["action_diff_mean_res2"]
                    action_diff_max += loss_dict["action_diff_max"]
            print(f"Test loss: {total_loss / len(val_loader)}")
            wandb.log({"eval/epoch_wise_action_diff": action_diff})
            wandb.log({"eval/epoch_wise_action_diff_tot": action_diff_tot})
            wandb.log({"eval/epoch_wise_action_diff_mean_res1": action_diff_mean_res1})
            wandb.log({"eval/epoch_wise_action_diff_mean_res2": action_diff_mean_res2})
            wandb.log({"eval/epoch_wise_action_diff_max": action_diff_max})

        
        for data in tqdm.tqdm(train_loader):
            # if epoch < (cfg.epochs * 0.5):
            optimizer["optimizer1"].zero_grad()
            optimizer["optimizer2"].zero_grad()
            # else:
            #     optimizer["optimizer2"].zero_grad()
            act = data["action"].to(cfg.device)
            rgb = data["rgb"].to(cfg.device)
            depth = data["depth"].to(cfg.device)
            goal = None
            if multi_gpu:
                predicted_act, loss, loss_dict = cbet_model.module(rgb, depth, goal, act)
            else:
                predicted_act, loss, loss_dict = cbet_model(rgb, depth, goal, act)
            wandb.log({"train/{}".format(x): y for (x, y) in loss_dict.items()})
            loss.backward()
            # if epoch < (cfg.epochs * 0.5):
            optimizer["optimizer1"].step()
            optimizer["optimizer2"].step()
            # else:
            #     optimizer["optimizer2"].step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")
        
        if epoch % cfg.save_every == 0:
            if multi_gpu:
                cbet_model.module.save_model(save_path)
            else:
                cbet_model.save_model(save_path)

if __name__ == "__main__":
    main()
