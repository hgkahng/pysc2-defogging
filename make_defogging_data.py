# -*- coding: utf-8 -*-

import os
import io
import json
import math
import enum
import pathlib
import zipfile

from multiprocessing import Pool
from functools import partial

import tqdm
import numpy as np
import scipy.sparse as sp

from absl import app
from absl import flags
from absl import logging
from pysc2.lib.units import Neutral, Protoss, Terran, Zerg
from pysc2.lib.static_data import UNIT_TYPES
from pysc2.lib.features import PlayerRelative
from pysc2.lib.named_array import NamedNumpyArray
from features.custom_features import SPATIAL_FEATURES


FLAGS = flags.FLAGS
flags.DEFINE_string('replay_root', default='./', help='')
flags.DEFINE_string('write_root', default='./data/defogging_/', help='')
flags.DEFINE_integer('size', default=32, help='')
flags.DEFINE_integer('processes', default=12, help='')
flags.DEFINE_integer('resume_from', default=0, help='')
flags.DEFINE_integer('min_game_length', default=3000, help='Game length lower bound.')
flags.DEFINE_bool('include_neutral', default=False, help='')
logging.set_verbosity(logging.INFO)

# print(FLAGS)


class Placeholder(object):
    def __init__(self, race: enum.IntEnum, size: int = 32):

        self.shape = (len(race), size, size)
        self.names = [u.name for u in race]

        self.data = NamedNumpyArray(
            np.zeros(self.shape, dtype=np.int32),
            names=[self.names, None, None]
        )

    def increment(self, name: str, i: int, j: int):
        self.data[name][i, j] += 1


def check_unit_type():
    for unit_id in UNIT_TYPES:
        try:
            _, _ = get_unit_type(unit_id)
        except ValueError:
            return False
    return True


def get_unit_type(unit_id: int, return_race: bool = True):
    """
    Customized version of `pysc2.lib.units.get_unit_type` function.
    This also returns the race as well as the unit name.
    """
    for race in (Neutral, Protoss, Terran, Zerg):
        try:
            unit_type = race(unit_id)
        except ValueError:
            continue
        if return_race:
            return unit_type, race
        else:
            return unit_type

    raise ValueError


def get_unit_types(unit_ids: list or tuple, return_race: bool = True):
    return [get_unit_type(uid, return_race=return_race) for uid in unit_ids]  # list of tuples; (unit_type, race)


def load_spatial_features(filename: str, size: int = None):
    if not os.path.basename(filename).endswith('.zip'):
        raise ValueError

    def load_feature(archive: object, name: str):
        """Load a feature."""
        if name not in SPATIAL_FEATURES._fields:
            raise KeyError("Not a supported feature.")
        i, feature = 0, []
        while True:
            try:
                with io.BufferedReader(archive.open(f"temp/{name}/{i}.npz", mode="r")) as f:
                    feature += [sp.load_npz(f)]
                    i += 1
            except KeyError:
                break
        return feature

    with zipfile.ZipFile(filename, 'r') as archive:
        unit_type = load_feature(archive, name='unit_type')
        player_relative = load_feature(archive, name='player_relative')

    return {
        'unit_type': unit_type,             # list of `scipy.sparse.coo_matrix`s
        'player_relative': player_relative  # list of `scipy.sparse.coo_matrix`s
    }


def process_unit_type(arr: sp.coo_matrix, size: int, race_p1: enum.IntEnum, race_p2: enum.IntEnum, include_neutral: bool = True):
    """
    Create a dictionary of replay placeholders, where each placeholder has 3D named numpy arrays.
    Arguments:
        arr: list of `scipy.sparse.coo_matrix`s.
        size: int.
        race_p1: `enum.IntEnum`, race of player 1, from `pysc2.lib.units.py`.
        race_p2: `enum.IntEnum`, race of player 2, from `pysc2.lib.units.py`.
    """
    if not math.log2(size).is_integer():
        raise ValueError("Not an appropriate size.")

    counts = {
        race_p1: Placeholder(race_p1, size),
        race_p2: Placeholder(race_p2, size),
    }

    if include_neutral:
        counts[Neutral] = Placeholder(Neutral, size)

    h, w = arr.shape
    dh, dw = h // size, w // size

    # Update unit counts
    for r, c, d in zip(arr.row, arr.col, arr.data):

        new_r = r // dh  # new row index
        new_c = c // dw  # new column index

        unit_type, race = get_unit_type(d, return_race=True)
        if race not in counts:
            counts[race] = Placeholder(race, size)  # redundant
        counts[race].increment(unit_type.name, new_r, new_c)

    return counts


def get_units(d: dict, include_neutral: bool = True):
    """Add function docstring."""
    if not 'player_relative' in d:
        raise ValueError
    if not 'unit_type' in d:
        raise ValueError

    if isinstance(d['player_relative'], list):
        assert all(isinstance(m, sp.coo_matrix) for m in d['player_relative'])

        if isinstance(d['unit_type'], list):
            assert all(isinstance(m, sp.coo_matrix) for m in d['unit_type'])

        units = []
        for pr, u in zip(d['player_relative'], d['unit_type']):
            pr_arr = pr.toarray()
            mask_self = pr_arr == PlayerRelative.SELF
            mask_enemy = pr_arr == PlayerRelative.ENEMY
            mask = mask_self + mask_enemy
            if include_neutral:
                mask += pr_arr == PlayerRelative.NEUTRAL
            units += [sp.coo_matrix(u.toarray() * mask)]
    else:
        mask_self = d['player_relative'] == PlayerRelative.SELF
        mask_enemy = d['player_relative'] == PlayerRelative.ENEMY
        mask = mask_self + mask_enemy
        if include_neutral:
            mask += d['player_relative'] == PlayerRelative.NEUTRAL
        units = d['unit_type'] * mask

    return units


def main(argv):

    del argv

    if not os.path.exists(FLAGS.replay_root):
        raise NotADirectoryError

    matchup = pathlib.Path(FLAGS.replay_root).name  # TvP, TvZ, ...
    if matchup not in ['TvP']:
        raise NotADirectoryError(f"Invalid matchup: `{matchup}`.")

    token2race = {
        'T': Terran,
        'P': Protoss,
        'Z': Zerg,
        'N': Neutral,
    }
    race2token = {r: t for t, r in token2race.items()}
    p1_race, p2_race = [token2race[t] for t in matchup.split('v')]

    skipped = 0
    replays = sorted(os.listdir(FLAGS.replay_root))


    with tqdm.tqdm(total=len(replays), dynamic_ncols=True) as pbar:
        for i, replay in enumerate(replays, 1):

            if i < FLAGS.resume_from:
                skipped += 1
                pbar.update(1)
                continue

            replay_dir = os.path.join(FLAGS.replay_root, replay)

            with open(os.path.join(replay_dir, 'ReplayMetaInfo.json'), 'r') as f:
                game_length = json.load(f)['game_duration_loops']
                if game_length < FLAGS.min_game_length:
                    skipped += 1
                    pbar.update(1)
                    continue

            try:
                features_p1 = load_spatial_features(os.path.join(replay_dir, "SpatialFeatures_Player1.zip"))
                features_gt = load_spatial_features(os.path.join(replay_dir, "SpatialFeatures_Observer.zip"))
            except FileNotFoundError:
                skipped += 1
                pbar.update(1)
                continue

            p1_units = get_units(features_p1, include_neutral=FLAGS.include_neutral)  # player 1 (foggy)
            gt_units = get_units(features_gt, include_neutral=FLAGS.include_neutral)  # observer (no fog)

            def process_replay(p1_units: list, gt_units: list, p1_race, p2_race, include_neutral: bool = True):
                """Returns a dictionary including named 4D numpy arrays."""
                kwargs = dict(
                    size=FLAGS.size,
                    race_p1=p1_race,
                    race_p2=p2_race,
                    include_neutral=include_neutral
                )

                with Pool(FLAGS.processes) as p:
                    p1_counts = p.map(partial(process_unit_type, **kwargs), p1_units)

                with Pool(FLAGS.processes) as p:
                    gt_counts = p.map(partial(process_unit_type, **kwargs), gt_units)

                def named_stack(l: list):
                    names = [None] + [[n for n, i in l[0]._index_names[0].items()]] + [None, None]  # pylint: disable=protected-access
                    stacked = np.stack(l, axis=0)
                    return NamedNumpyArray(stacked, names=names)

                # FIXME: names are never saved when using `np.savez_compressed`. Save explicitly
                result = {
                    "input." + race2token[p1_race]: named_stack([phs[p1_race].data for phs in p1_counts]),
                    "input." + race2token[p2_race]: named_stack([phs[p2_race].data for phs in p1_counts]),
                }

                result.update(
                    {
                        "target." + race2token[p1_race]: named_stack([phs[p1_race].data for phs in gt_counts]),
                        "target." + race2token[p2_race]: named_stack([phs[p2_race].data for phs in gt_counts]),
                    }
                )

                if include_neutral:
                    result.update(
                        {
                            "input." + race2token[Neutral]: named_stack([phs[Neutral].data for phs in p1_counts]),
                            "target." + race2token[Neutral]: named_stack([phs[Neutral].data for phs in gt_counts]),
                        }
                    )

                result.update(
                    {
                        f"names.{race2token[p1_race]}": np.asarray([u.name for u in p1_race]),
                        f"names.{race2token[p2_race]}": np.asarray([u.name for u in p2_race]),
                    }
                )

                if include_neutral:
                    result.update(
                        {
                            f"names.{race2token[Neutral]}": np.asarray([u.name for u in Neutral]),
                        }
                    )

                return result

            # Create model input (first player's perspective of game)
            try:
                result = process_replay(p1_units, gt_units, p1_race, p2_race, FLAGS.include_neutral)
            except ValueError:
                skipped += 1
                pbar.update(1)
                continue

            write_dir = os.path.join(FLAGS.write_root, matchup, replay)
            os.makedirs(write_dir, exist_ok=True)

            np.savez_compressed(
                file=os.path.join(write_dir, f'data.defogging.{FLAGS.size}.{matchup}.npz'),
                **result
            )

            pbar.set_description_str(f"[{i:>05}/{len(replays):>05}]")
            pbar.update(1)

    print(f"Finished. {skipped:,} replays have been skipped.")

if __name__ == '__main__':
    if check_unit_type():
        app.run(main)
