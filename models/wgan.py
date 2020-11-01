# -*- coding: utf-8 -*-

"""..."""

import os
import collections
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from layers.base import Flatten


class Generator(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 **kwargs):  # pylint: disable=unused-argument
        super(Generator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ConvCritic(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 4, **kwargs):
        super(ConvCritic, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.layers = nn.Sequential()
        self.layers.add_module(
            'conv', self.make_layers(
                self.in_channels,
                self.out_channels,
                num_convs=kwargs.get('num_convs', 1),
                num_blocks=self.num_blocks
                )
            )
        self.layers.add_module("gap", nn.AdaptiveAvgPool2d(1))
        self.layers.add_module("flatten", Flatten())
        self.layers.add_module("linear", nn.Linear(self.out_channels, 1))

    def forward(self, x):
        return self.layers(x)

    @classmethod
    def make_layers(cls, in_channels: int, out_channels: int, num_convs: int = 1, num_blocks: int = 4):
        """
        Arguments:
            in_channels: int,
            out_channels: int,
            num_convs: int, number of convolutional layers within each block.
            num_blocks: int, number of convolutional blocks
        """

        layers = []

        # Second to last layers
        for _ in range(num_blocks - 1):
            blk_kwargs = dict(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                n=num_convs,
                activation='lrelu'
            )
            layers = [cls.make_block(**blk_kwargs)] + layers  # insert to the front (LIFO)
            out_channels = out_channels // 2

        # First layer
        first_blk_kwargs = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            n=num_convs,
            activation='lrelu'
        )
        layers = [cls.make_block(**first_blk_kwargs)] + layers

        return nn.Sequential(*layers)

    @staticmethod
    def make_block(in_channels: int, out_channels: int, n: int = 1, activation: str = 'lrelu'):
        if activation == 'relu':
            act_fn = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            act_fn = nn.LeakyReLU(.2, inplace=True)
        else:
            raise NotImplementedError

        block = nn.Sequential()

        strides = [2] + [1] * (n - 1)
        for i, s in enumerate(strides):
            conv_kwargs = dict(kernel_size=4, stride=s, padding=s-1)
            block.add_module(f"conv{i}", nn.Conv2d(in_channels, out_channels, **conv_kwargs))
            block.add_module(f"bnorm{i}", nn.BatchNorm2d(out_channels))
            block.add_module(f"act{i}", act_fn)
            in_channels = out_channels

        return block


class WGAN_GP(object):
    """Add class docstring."""
    def __init__(self,
                 G: nn.Module,
                 C: nn.Module,
                 G_opt: optim.Optimizer,
                 C_opt: optim.Optimizer,
                 **kwargs):  # pylint: disable=unused-argument

        super(WGAN_GP, self).__init__()

        self.G = G
        self.C = C
        self.G_opt = G_opt
        self.C_opt = C_opt
        self.lambda_gp = kwargs.get("lambda_gp", 10.)
        self.n_critic_updates = kwargs.get("n_critic_updates", 5)

        if kwargs.get('use_reconstruction_loss', False):
            self.pixelwise_loss = nn.MSELoss(reduction='mean')
        else:
            self.pixelwise_loss = None

        self.initialize_weights(self.G)
        self.initialize_weights(self.C)

    def train(self, data_loader, device: str, **kwargs):

        self.G.to(device)
        self.C.to(device)

        self.G.train()
        self.C.train()

        train_G_loss = 0
        train_C_loss = 0
        steps_per_epoch = len(data_loader)

        with tqdm.tqdm(total=steps_per_epoch, leave=False, dynamic_ncols=True) as pbar:
            for i, batch in enumerate(data_loader):

                x_real = batch['input'].to(device)

                # 1. Train critic (a.k.a discriminator)
                x_fake = self.G(x_real)

                # y_real = torch.ones(x_real.size(0), 1, device=device).float()
                # y_fake = torch.zeros_like(y_real)

                gradient_penalty = self.compute_gradient_penalty(
                    critic=self.C,
                    real=x_real,
                    fake=x_fake,
                    device=device,
                )

                c_loss =  \
                    - torch.mean(self.C(x_real)) \
                    + torch.mean(self.C(x_fake)) \
                    + self.lambda_gp * gradient_penalty

                c_loss.backward()
                self.C_opt.step()
                self.C_opt.zero_grad()
                train_C_loss += c_loss.item()

                # 2. Train generator (encoder-decoder)
                if i % self.n_critic_updates == 0:

                    x_fake = self.G(x_real)
                    g_loss = - torch.mean(self.C(x_fake))

                    if self.pixelwise_loss is not None:
                        pxl_loss = self.pixelwise_loss(x_fake, x_real)
                        g_loss = 0.01 * g_loss + 0.99 * pxl_loss
                        # g_loss = 0.5 * (g_loss + pxl_loss)

                    g_loss.backward()
                    self.G_opt.step()
                    self.G_opt.zero_grad()

                    train_G_loss += g_loss.item()

                pbar.update(1)

        out = {
            "C_loss": train_C_loss / steps_per_epoch,
            "G_loss": train_G_loss / (steps_per_epoch / self.n_critic_updates),
        }
    
        return out

    def save_checkpoint(self, path: str):
        ckpt_dir = os.path.dirname(path)
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt = {
            'G': self.G.state_dict(),
            'C': self.C.state_dict(),
            'G_opt': self.G_opt.state_dict(),
            'C_opt': self.C_opt.state_dict()
        }
        torch.save(ckpt, path)

    @staticmethod
    def initialize_weights(model: nn.Module):
        for _, m in model.named_modules():

            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, mean=.0, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def compute_gradient_penalty(critic: nn.Module,
                                 real: torch.Tensor,
                                 fake: torch.Tensor,
                                 device: str):
        """Function for calculating gradient penalties."""

        assert real.ndim == fake.ndim == 4
        assert real.size() == fake.size()

        batch_size = real.size(0)

        # Random weights for interpolation between real & fake
        alpha = torch.rand(batch_size, 1, 1, 1, requires_grad=False, device=device)

        # Interpolation between real & fake
        interpolates = alpha * real + (1 - alpha) * fake
        assert interpolates.requires_grad

        y_fake = torch.ones(batch_size, 1, requires_grad=False, device=device)

        # Get gradients with respect to interpolates
        gradients = torch.autograd.grad(
            outputs=critic(interpolates),
            inputs=interpolates,
            grad_outputs=y_fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = (gradients.norm(2, dim=-1) - 1) ** 2
        gradient_penalty = gradient_penalty.mean()

        return gradient_penalty
