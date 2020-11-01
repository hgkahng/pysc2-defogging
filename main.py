# -*- coding: utf-8 -*-

"""Main training script."""

import os
import datetime

import torch.nn as nn
import torch.optim as optim

from absl import app
from absl import flags
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console

from layers.encoders import Encoder
from layers.decoders import Decoder
from models.wgan import WGAN_GP, ConvCritic
from utils.data import DefoggingDataset

FLAGS = flags.FLAGS
flags.DEFINE_enum('model', default='wgan-gp', help='', enum_values=['aae', 'wgan-gp'])
flags.DEFINE_string('data_dir', default='./data/defogging', help='')
flags.DEFINE_string('matchup', default='TvP', help='')
flags.DEFINE_integer('size', default=32, help='')
flags.DEFINE_integer('epochs', default=1000, help='')
flags.DEFINE_integer('batch_size', default=256, help='')
flags.DEFINE_integer('num_workers', default=8, help='')
flags.DEFINE_string('device', default='cuda:0', help='')
flags.DEFINE_integer('latent_dim', default=100, help='')
flags.DEFINE_bool('reparameterize', default=False, help='')
flags.DEFINE_string('optimizer', default='adamw', help='')
flags.DEFINE_float('learning_rate', default=1e-4, help='')
flags.DEFINE_float('weight_decay', default=5e-5, help='')


def main(argv=None):

    del argv
    console = Console()

    dataset = DefoggingDataset(root_dir=FLAGS.data_dir, size=FLAGS.size, matchup=FLAGS.matchup)

    split_idx = int(len(dataset) * 0.99)
    train_indices = list(range(len(dataset)))[:split_idx]
    # _ = list(range(len(dataset)))[split_idx:] * 2
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=FLAGS.num_workers,
        pin_memory=True,
        drop_last=False
    )

    ckpt_dir = f"./checkpoints/defogging/{FLAGS.model}/" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    writer = SummaryWriter(log_dir=ckpt_dir)

    input_shape = (60 + 45, 32, 32)  # T:60, P: 45

    # Generator
    encoder = Encoder(
        input_shape=input_shape,
        out_channels=512,
        latent_dim=FLAGS.latent_dim,
        reparameterize=FLAGS.reparameterize,
    )
    decoder = Decoder(
        in_channels=512,
        output_shape=input_shape,
        latent_dim=FLAGS.latent_dim,
    )

    # Discriminator
    if FLAGS.model == 'wgan-gp':
        critic = ConvCritic(
            in_channels=input_shape[0],
            out_channels=512
            )
    elif FLAGS.model == 'aae':
        raise NotImplementedError
    else:
        raise NotImplementedError

    # Optimizers
    if FLAGS.optimizer == 'adamw':
        g_opt = optim.AdamW(
            params=[
                {'params': encoder.parameters()},
                {'params': decoder.parameters()}
            ],
            lr=FLAGS.learning_rate,
            weight_decay=FLAGS.weight_decay,
            betas=(.5, .9),
        )
        c_opt = optim.AdamW(
            params=[
                {'params': critic.parameters()},
            ],
            lr=FLAGS.learning_rate,
            weight_decay=FLAGS.weight_decay,
            betas=(.5, .9),
        )

    if FLAGS.model == 'wgan-gp':

        model = WGAN_GP(
            G=nn.Sequential(encoder, decoder),
            C=critic,
            G_opt=g_opt,
            C_opt=c_opt,
            use_reconstruction_loss=False
        )
    else:
        raise NotImplementedError

    for epoch in range(1, FLAGS.epochs + 1):
        train_history = model.train(train_loader, alpha=0.01, device='cuda')
        console.print(f"Train [{epoch:>04}]/[{FLAGS.epochs:>04}]: ", end='', style="Bold Cyan")
        console.print(*[f"{k}:{v:.4f}" for k, v in train_history.items()], sep=' | ', style='Bold Blue')
        for k, v in train_history.items():
            writer.add_scalar(tag=k, scalar_value=v, global_step=epoch)

        if epoch % 10 == 0:
            model.save_checkpoint(os.path.join(ckpt_dir, f"model.epoch_{epoch:04d}.pt"))

    model.save_checkpoint(os.path.join(ckpt_dir, "model.final.pt"))


if __name__ == '__main__':
    app.run(main)
