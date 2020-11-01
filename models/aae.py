# -*- coding: utf-8 -*-

"""Adversarial autoencoders."""

import os
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from layers.encoders import Encoder
from layers.decoders import Decoder


class AAE(object):
    def __init__(self,
                 input_shape: tuple,
                 enc_out_channels: int,
                 latent_dim: int = 10,
                 **kwargs):

        self.input_shape = self.output_shape = input_shape
        self.enc_out_channels = self.dec_in_channels = enc_out_channels
        self.latent_dim = latent_dim
        self.num_blocks = kwargs.get('num_blocks', 4)
        self.reparameterize = kwargs.get('reparameterize', True)

        self.encoder = Encoder(
            input_shape=self.input_shape,
            out_channels=self.enc_out_channels,
            latent_dim=self.latent_dim,
            num_blocks=self.num_blocks,
        )
        self.decoder = Decoder(
            in_channels=self.dec_in_channels,
            output_shape=self.output_shape,
            latent_dim=self.latent_dim,
            num_blocks=self.num_blocks
        )
        self.critic = DenseCritic(in_features=latent_dim)

        self.g_opt = optim.AdamW(
            params=[
                {'params': self.encoder.parameters()},
                {'params': self.decoder.parameters()},
            ],
            lr=1e-3,
            weight_decay=1e-4,
            betas=(0.9, 0.99),
        )

        self.c_opt = optim.AdamW(
            params=[
                {'params': self.critic.parameters()}
            ],
            lr=1e-3,
            weight_decay=1e-4,
            betas=(0.5, 0.99)
        )

        self.pixelwise_loss = nn.L1Loss(reduction='mean')
        self.adversarial_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def train(self, data_loader, alpha: 0.01, device: str = 'cuda:0'):

        self.encoder.to(device)
        self.decoder.to(device)
        self.critic.to(device)

        self.encoder.train()
        self.decoder.train()
        self.critic.train()

        train_G_adv_loss = train_G_recon_loss = 0
        train_C_fake_loss = train_C_real_loss = 0
        steps_per_epoch = len(data_loader)

        with tqdm.tqdm(total=steps_per_epoch, leave=False, dynamic_ncols=False) as pbar:
            for _, batch in enumerate(data_loader):

                enc_input = batch['input'].to(device)
                dec_target = batch['target'].to(device)
                batch_size = enc_input.size(0)

                y_real = torch.ones(batch_size, 1, device=device, requires_grad=False).float()  # 1s
                y_fake = torch.zeros_like(y_real)                                               # 0s

                # 1. Optimize generator
                self.g_opt.zero_grad()
                z_fake = self.encoder(enc_input)
                dec_output = self.decoder(z_fake)
                g_adv_loss = self.adversarial_loss(self.critic(z_fake), y_real)
                g_recon_loss = self.pixelwise_loss(dec_output, dec_target)
                g_loss = alpha * g_adv_loss + (1-alpha) * g_recon_loss
                g_loss.backward()
                # nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.)
                # nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.)
                self.g_opt.step()

                # 1. Optimize critic
                self.c_opt.zero_grad()
                z_fake = self.encoder(enc_input)
                dec_output = self.decoder(z_fake)
                z_real = Normal(loc=torch.zeros_like(z_fake), scale=1.).sample()
                c_fake_loss = self.adversarial_loss(self.critic(z_fake), y_fake)
                c_real_loss = self.adversarial_loss(self.critic(z_real), y_real)
                c_loss = 0.5 * (c_fake_loss + c_real_loss)
                c_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.)
                self.c_opt.step()

                train_G_adv_loss += g_adv_loss.item()
                train_G_recon_loss += g_recon_loss.item()
                train_C_fake_loss += c_fake_loss.item()
                train_C_real_loss += c_real_loss.item()

                pbar.update(1)

        return {
            'G_adv_loss': train_G_adv_loss / steps_per_epoch,
            'G_recon_loss': train_G_recon_loss / steps_per_epoch,
            'C_fake_loss': train_C_fake_loss / steps_per_epoch,
            'C_real_loss': train_C_real_loss / steps_per_epoch,
        }

    def evaluate(self, data_loader):
        raise NotImplementedError

    def save_checkpoint(self, path: str):
        ckpt_dir = os.path.dirname(path)
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt = {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'critic': self.critic.state_dict(),
            'g_opt': self.g_opt.state_dict(),
            'c_opt': self.c_opt.state_dict()
        }
        torch.save(ckpt, path)


class DenseCritic(nn.Module):
    def __init__(self, in_features: int):
        super(DenseCritic, self).__init__()
        self.in_features = in_features
        self.layers = nn.Sequential(
            nn.Linear(self.in_features, self.in_features),
            nn.BatchNorm1d(self.in_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.in_features, self.in_features // 2),
            nn.BatchNorm1d(self.in_features // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.in_features // 2, 1)
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)  # [B, 1]




if __name__ == "__main__":

    with torch.no_grad():
        batch_shape = (64, 100, 32, 32)
        encoder = Encoder(input_shape=batch_shape[1:], out_channels=512, latent_dim=10, num_blocks=4)
        enc = encoder(torch.zeros(*batch_shape))
        decoder = Decoder(in_channels=512, output_shape=batch_shape[1:], latent_dim=10, num_blocks=4)
        dec = decoder(enc)

        print(enc.shape)
        print(dec.shape)
