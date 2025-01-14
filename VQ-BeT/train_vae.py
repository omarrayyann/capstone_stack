import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import tqdm
from torch.utils.data import DataLoader
import wandb

@hydra.main(version_base=None, config_path="config", config_name="jubail")
def main(cfg: DictConfig):

    # Initialize WandB
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    save_path = Path(cfg.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Instantiate VQ-VAE model and dataset
    vqvae_model = instantiate(cfg.vqvae)
    dataset = instantiate(cfg.dataset)

    # Split dataset and create data loaders
    train_dataset, val_dataset = instantiate(cfg.split_dataset, dataset=dataset)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    for epoch in tqdm.trange(cfg.train.epochs):
        epoch_loss = 0.0

        for data in tqdm.tqdm(train_loader):
            actions = data["action"].to(cfg.device)

            encoder_loss, vq_loss_state, vq_code, vqvae_recon_loss = vqvae_model.vqvae_update(actions)
            batch_loss = encoder_loss + vq_loss_state + vqvae_recon_loss
            epoch_loss += batch_loss.item()

        # Log metrics to WandB
        wandb.log({
            "epoch": epoch,
            "encoder_loss": encoder_loss.item(),
            "vq_loss_state": vq_loss_state.item(),
            "reconstruction_loss": vqvae_recon_loss.item(),
            "epoch_loss": epoch_loss,
        })

        print(f"Epoch {epoch}, Encoder Loss: {encoder_loss}, VQ Loss: {vq_loss_state}, Recon Loss: {vqvae_recon_loss}")

        if epoch % cfg.train.save_interval == 0:
            state_dict = vqvae_model.state_dict()
            torch.save(state_dict, os.path.join(save_path, "trained_vqvae.pt"))

    wandb.finish()


if __name__ == "__main__":
    main()
