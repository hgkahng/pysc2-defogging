# -*- coding: utf-8 -*-

"""..."""

import os
import json
import time

import numpy as np

from functools import partial
from multiprocessing import Pool
from absl import app
from absl import flags
from rich.console import Console


FLAGS = flags.FLAGS
flags.DEFINE_string("root", default="./data/defogging/", help="")
flags.DEFINE_string("matchup", default="TvP", help="")
flags.DEFINE_integer("size", default=32, help="")
flags.DEFINE_integer('processes', default=12, help='')


def save_timestep_pair(t: int, inp: np.ndarray, tar: np.ndarray, write_dir: str, names: np.ndarray):
    """Use multiprocessing!"""

    os.makedirs(write_dir, exist_ok=True)
    write_file = os.path.join(write_dir, f"{t}.npz")
    if os.path.exists(write_file):
        os.remove(write_file)
    np.savez_compressed(write_file, input=inp[t], target=tar[t], names=names)


def main(argv):
    """Main function."""

    del argv
    console = Console()

    root = os.path.join(FLAGS.root, FLAGS.matchup)
    if not os.path.isdir(root):
        raise NotADirectoryError

    replay_dirs = sorted(os.listdir(root))
    replay_dirs = [os.path.join(root, rd) for rd in replay_dirs]
    num_replays = len(replay_dirs)

    p1, p2 = FLAGS.matchup.split("v")  # TvP -> [T, P]

    skipped = []
    for i, replay_dir in enumerate(replay_dirs):

        src_npz = os.path.join(replay_dir, f"data.defogging.{FLAGS.size}.{FLAGS.matchup}.npz")

        if not os.path.isfile(src_npz):
            skipped.append(replay_dir)
            console.print(f"[{i:05}/{num_replays:05}]: {src_npz} does not exist. Skipping.", style="Bold Red")
            continue

        # Open meta file (ReplayMetaInfo.json)
        meta_file = os.path.join(replay_dir, "ReplayMetaInfo.json")
        with open(meta_file, 'r') as f:
            _ = json.load(f)

        # Load numpy data; 4D arrays of shape (T, C, H, W)
        with np.load(src_npz) as src:
            inp_p1 = src[f"input.{p1}"]
            inp_p2 = src[f"input.{p2}"]
            tar_p1 = src[f"target.{p1}"]
            tar_p2 = src[f"target.{p2}"]
            names_p1 = src[f"names.{p1}"]
            names_p2 = src[f"names.{p2}"]

        # Concatenate along the dimension 1 (`unit_type` dimension)
        inp = np.concatenate([inp_p1, inp_p2], axis=1)
        tar = np.concatenate([tar_p1, tar_p2], axis=1)
        names = np.concatenate([names_p1, names_p2], axis=-1)

        # Slice and save along dimension 0 (`time` dimension)
        if inp.shape != tar.shape:
            raise ValueError(f"The shape of input ({inp.shape}) and target ({tar.shape}) do not match")

        start = time.time()
        write_dir = os.path.join(replay_dir, f"{FLAGS.size}x{FLAGS.size}")
        os.makedirs(write_dir, exist_ok=True)

        timesteps = list(range(len(inp)))
        with Pool(FLAGS.processes) as p:
            p.map(partial(save_timestep_pair, inp=inp, tar=tar, names=names, write_dir=write_dir), timesteps)
        end = time.time()
        console.print(f"[{i+1:05}/{num_replays:05}]: {len(timesteps)} frames, {end - start:.3f} seconds.", style="Bold Green")


if __name__ == '__main__':
    app.run(main)
