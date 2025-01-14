import os
import random
from pathlib import Path
from torch.utils.data import DataLoader
import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf
from vqvae.vqvae import *
import wandb


def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)


@hydra.main(config_path="configs", config_name="pre_train", version_base="1.2")
def main(cfg):
    run = wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    save_path = Path(cfg.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    vqvae_model = hydra.utils.instantiate(cfg.vqvae_model)
    
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    dataset = hydra.utils.instantiate(cfg.dataset)
    train_dataset, val_dataset = hydra.utils.instantiate(cfg.split_dataset, dataset=dataset)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    for epoch in tqdm.trange(cfg.epochs):
        for data in tqdm.tqdm(train_loader):
            act = data["action"].to(cfg.device)[:,-1,:].reshape(-1, 1, 8)

            (
                encoder_loss,
                vq_loss_state,
                vq_code,
                vqvae_recon_loss,
            ) = vqvae_model.vqvae_update(act)  # N T D

            wandb.log({"pretrain/n_different_codes": len(torch.unique(vq_code))})
            wandb.log(
                {"pretrain/n_different_combinations": len(torch.unique(vq_code, dim=0))}
            )
            wandb.log({"pretrain/encoder_loss": encoder_loss})
            wandb.log({"pretrain/vq_loss_state": vq_loss_state})
            wandb.log({"pretrain/vqvae_recon_loss": vqvae_recon_loss})
        print(
            f"Epoch {epoch}, Encoder Loss: {encoder_loss}, VQ Loss: {vq_loss_state}, Recon Loss: {vqvae_recon_loss}"
        )
        if epoch % 20 == 0:
            state_dict = vqvae_model.state_dict()
            torch.save(state_dict, os.path.join(save_path, "trained_vqvae.pt"))


if __name__ == "__main__":
    main()
