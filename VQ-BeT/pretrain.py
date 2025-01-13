import os
import random
from pathlib import Path

import numpy as np
import torch
import tqdm
from vqvae.vqvae import *

def main():
  
    save_path = Path("/home/franka/Desktop/franka_stack/VQ-BeT/saved")
    save_path.mkdir(parents=True, exist_ok=False)

    vqvae_model = VqVae(
        obs_dim=99, # unused
        input_dim_h=15,# length of action chunk
        input_dim_w=8, # action dim
        n_latent_dims=512,
        vqvae_n_embed=32,
        vqvae_groups=4,
        eval=True,
        device="cuda",
        load_dir=None,
        encoder_loss_multiplier=1.0,
        act_scale=1.0,
    )
    
    
    
    hydra.utils.instantiate(cfg.vqvae_model)
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    train_data, test_data = hydra.utils.instantiate(cfg.data)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.batch_size, shuffle=True, pin_memory=False
    )
    for epoch in tqdm.trange(cfg.epochs):
        for data in tqdm.tqdm(train_loader):
            obs, act, goal = (x.to(cfg.device) for x in data)

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

        if epoch % 50 == 0:
            state_dict = vqvae_model.state_dict()
            torch.save(state_dict, os.path.join(save_path, "trained_vqvae.pt"))


if __name__ == "__main__":
    main()
