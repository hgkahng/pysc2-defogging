# -*- coding: utf-8 -*-

"""
    Parsing replays.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import glob
import json
import time
import websocket

from absl import app
from absl import flags
from absl import logging
from pysc2 import run_configs
from pysc2.env import environment
from pysc2.lib import point
from pysc2.lib import protocol
from pysc2.lib import gfile
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import common_pb2 as sc_common
from rich.console import Console

from parsers import ParserBase, SpatialFeatParser
from features.custom_features import custom_features_from_game_info


FLAGS = flags.FLAGS
flags.DEFINE_string('sc2_path', default='C:/Program Files (x86)/StarCraft II/', help='Path to SC2 client.')
flags.DEFINE_string('map_dir', default='C:/Program Files (x86)/StarCraft II/Maps/', help='Directory containing maps.')
flags.DEFINE_string('replay_file', default=None, help='Replay file, optional if replay directory is not specified.')
flags.DEFINE_string('replay_dir', default=None, help='Replay directory, optional if replay file is not specified.')
flags.DEFINE_string('result_dir', default='./parsed/', help='Directory to write parsed files.')
flags.DEFINE_integer('screen_size', default=64, help='Size of game screen.')
flags.DEFINE_integer('minimap_size', default=64, help='Size of minimap.')
flags.DEFINE_integer('step_mul', default=4, help='Sample interval.')
flags.DEFINE_integer('min_game_length', default=3000, help='Game length lower bound.')
flags.DEFINE_integer('resume_from', default=0, help='Index of replay to resume from.')
flags.DEFINE_float('discount', default=1., help='Not used.')
flags.DEFINE_bool('override', default=False, help='Force overriding existing results.')
flags.DEFINE_bool('only_info', default=False, help='Get only replay meta information.')
flags.DEFINE_list('player_id', default=[1, 2], help='From whom the game will be played.')

flags.DEFINE_enum(
    'race_matchup', default=None, help='Race match-ups.',
    enum_values=['all', 'TvT', 'TvP', 'TvZ', 'PvT', 'PvP', 'PvZ', 'ZvT', 'ZvP', 'ZvZ']
    )
flags.register_validator(
    'race_matchup',
    lambda matchup: matchup == 'all' or all([race in ['T', 'P', 'Z'] for race in matchup.split('v')])
    )

logging.set_verbosity(logging.INFO)


def check_flags():
    """Check validity of command line arguments."""
    if FLAGS.replay_file is not None and FLAGS.replay_dir is not None:
        raise ValueError("Only one of `replay_file` and `replay_dir` must be specified.")
    else:
        if FLAGS.replay_file is not None:
            logging.info("Parsing a single replay.")
        elif FLAGS.replay_dir is not None:
            logging.info("Parsing replays in {}".format(FLAGS.replay_dir))
        else:
            raise ValueError("Both `replay_file` and `replay_dir` are not specified.")


class ReplayRunner(object):
    """
    Parsing replay data, based on the following implementations:
        https://github.com/narhen/pysc2-replay/blob/master/transform_replay.py
        https://github.com/deepmind/pysc2/blob/master/pysc2/bin/replay_actions.py
    """
    def __init__(self,
                 replay_file: str,
                 player_id: list = [1],
                 screen_size: tuple = (64, 64),
                 minimap_size: tuple = (64, 64),
                 discount: float = 1.,
                 step_mul: int = 1,
                 override: bool = False,
                 **kwargs):  # pylint: disable=dangerous-default-value

        self.replay_file = os.path.abspath(replay_file)
        self.replay_name = os.path.split(replay_file)[-1].replace('.SC2Replay', '')

        self.player_id = player_id
        self.discount = discount
        self.step_mul = step_mul
        self.override = override

        # Configure screen size
        if isinstance(screen_size, tuple):
            self.screen_size = screen_size
        elif isinstance(screen_size, int):
            self.screen_size = (screen_size, screen_size)
        else:
            raise ValueError("Argument `screen_size` requires a tuple of size 2 or a single integer.")

        # Configure minimap size
        if isinstance(minimap_size, tuple):
            self.minimap_size = minimap_size
        elif isinstance(minimap_size, int):
            self.minimap_size = (minimap_size, minimap_size)
        else:
            raise ValueError("Argument `minimap_size` requires a tuple of size 2 or a single integer.")

        assert len(self.screen_size) == 2
        assert len(self.minimap_size) == 2

        # Arguments for 'sc_process.StarCraftProcess'. Check the following:
        # https://github.com/deepmind/pysc2/blob/master/pysc2/lib/sc_process.py
        try:
            sc2_process_configs = {"full_screen": False, 'timeout_seconds': 300}
            self.run_config = run_configs.get()            # class Windows(LocalBase) <- LocalBase(lib.RunConfig)
            self.sc2_process = self.run_config.start(**sc2_process_configs)
            self.controller = self.sc2_process.controller  # be sure to 'self.controller.close()' to cleanup
        except websocket.WebSocketTimeoutException as err:
            raise ConnectionRefusedError(f"Connection to SC2 process unavailable. ({err})")
        except protocol.ConnectionError as err:
            raise ConnectionRefusedError(f"Connection to SC2 process unavailable. ({err})")

        # Check the following links for usage of run_config and controller.
        #   https://github.com/deepmind/pysc2/blob/master/pysc2/run_configs/platforms.py
        #   https://github.com/deepmind/pysc2/blob/master/pysc2/lib/sc_process.py
        #   https://github.com/deepmind/pysc2/blob/master/pysc2/lib/remote_controller.py

        # Load replay information
        self.replay_data = self.run_config.replay_data(self.replay_file)  # read replay
        info = self.controller.replay_info(self.replay_data)              # get info

        # Check validity
        if not self.check_valid_replay(info, self.controller.ping()):
            self.safe_escape()
            raise ValueError("Replay invalid.")

        # Check total length
        if not self.check_valid_game_length(info, FLAGS.min_game_length):
            self.safe_escape()
            raise ValueError("Replay too short.")

        # Check match-up configuration
        if FLAGS.race_matchup != 'all':
            if not self.check_valid_matchup(info, matchup=FLAGS.race_matchup):
                self.safe_escape()
                raise ValueError("Replay match-up invalid.")

        # 'raw=True' enables the use of 'feature_units'
        # https://github.com/Blizzard/s2client-proto/blob/master/docs/protocol.md#interfaces
        self.interface = sc_pb.InterfaceOptions(
            raw=False,
            score=True,
            show_cloaked=False,
            feature_layer=sc_pb.SpatialCameraSetup(width=24, allow_cheating_layers=True)
        )

        self.screen_size = point.Point(*self.screen_size)
        self.minimap_size = point.Point(*self.minimap_size)
        self.screen_size.assign_to(self.interface.feature_layer.resolution)
        self.minimap_size.assign_to(self.interface.feature_layer.minimap_resolution)

        # Find map data
        self.map_name = info.map_name
        self.map_data = None
        if 'map_dir' in kwargs:
            map_dir = kwargs.get('map_dir', None)
            assert os.path.isdir(map_dir)
            try:
                self.map_data = self.get_map_data(
                    map_dir=map_dir,
                    map_name=self.map_name
                )
            except ValueError:
                logging.info("Proceeding without map data.")

        self._episode_length = info.game_duration_loops
        self._episode_steps = 0
        self._state = environment.StepType.FIRST  # default starting value, modified throughout the process
        self.info = info

    def _start_and_parse_replay(self, parsers: dict, player_id: int, disable_fog: bool):
        """Start and parse replay, from a specified player's perspective."""

        self.controller.start_replay(
            req_start_replay=sc_pb.RequestStartReplay(
                replay_data=self.replay_data,
                map_data=self.map_data,
                options=self.interface,
                observed_player_id=player_id,
                disable_fog=disable_fog,
            )
        )

        _features = custom_features_from_game_info(self.controller.game_info())

        while True:

            # Take a step, scale specified by 'step_mul' (sc_pb, RequestStep -> ResponseStep)
            self.controller.step(self.step_mul)

            # Receive observation (sc_pb, RequestObservation -> ResponseObservation)
            obs = self.controller.observe()

            # '.transform_obs' is defined under features.Features
            try:
                agent_obs = _features.custom_transform_obs(obs)
            except ValueError as err:
                # e.g. Unknown ability_id: 1, type: cmd_quick. Likely a bug.
                # This error is fixed in later version of SC2 clients (>=4.1.2)
                logging.info(f"{err}. Using previous observation.")

            if obs.player_result:
                self._state = environment.StepType.LAST
                discount = 0
            else:
                self._state = environment.StepType.MID
                discount = self.discount

            self._episode_steps += self.step_mul

            step_kwargs = dict(
                step_type=self._state,
                reward=0,
                discount=discount,
                observation=agent_obs
            )
            step = environment.TimeStep(**step_kwargs)

            if self._state == environment.StepType.MID:
                for parser in parsers:
                    parser.parse(timestep=step)
            elif self._state == environment.StepType.LAST:
                logging.info("Reached end of current game.")
                for parser in parsers:
                    parser.save(path=None, override=FLAGS.override)
                break
            else:
                pass

    def start(self, parser_objects: dict, sparse: bool = True, **kwargs):
        """
        Parse replays.
        Arguments:
            parsers: dict, name -> a subclass of `ParserBase` object
        """

        console = kwargs.get('console', None)

        for _, parser_obj in parser_objects.items():
            if not issubclass(parser_obj, ParserBase):
                raise ValueError

        # Create a directory to write to
        os.makedirs(self.write_dir, exist_ok=True)

        # Save replay meta information
        replay_meta_info = self.get_replay_meta_info(self.info)
        replay_meta_info['replay_file'] = self.replay_file
        meta_info_file = os.path.join(self.write_dir, 'ReplayMetaInfo.json')
        if not os.path.exists(meta_info_file):
            with open(meta_info_file, 'w') as fp:
                json.dump(replay_meta_info, fp, indent=4)
        if FLAGS.only_info:
            logging.info("Only retrieving replay meta information. Terminating.")
            return

        # Parse w/ observer perspective (disabling fog)
        parsers = []
        for name, parser_obj in parser_objects.items():
            parser = parser_obj(sparse=sparse)
            setattr(
                parser,
                'write_file',
                os.path.join(self.write_dir, f"{name}_Observer.zip")
            )
            parsers += [parser]
        if console is not None:
            console.print("Parsing from observer's perspective", style='bold cyan')
        self._start_and_parse_replay(
            parsers=parsers,
            player_id=1,
            disable_fog=True,
        )

        # Parse (single player perspective, p1 & p2)
        for player_id in self.player_id:
            parsers = []
            for name, parser_obj in parser_objects.items():
                parser = parser_obj(sparse=sparse)
                setattr(
                    parser,
                    'write_file',
                    os.path.join(self.write_dir, f"{name}_Player{player_id}.zip")
                )
                parsers += [parser]
            if console is not None:
                console.print(f"Parsing from player {player_id}'s perspective", style='bold green')
            self._start_and_parse_replay(
                parsers=parsers,
                player_id=player_id,
                disable_fog=False
            )

    @property
    def write_dir(self):
        """Directory to write results to."""
        return os.path.join(FLAGS.result_dir, self.get_matchup(self.info), self.replay_name)

    @staticmethod
    def check_valid_replay(info, ping):
        """Check validity of replay."""
        if info.HasField('error'):
            logging.info('Replay has error.')
            return False
        elif info.base_build != ping.base_build:
            logging.info('Replay from different base build.')
            return False
        elif len(info.player_info) != 2:
            logging.info('Replay not a game with two players.')
            return False
        else:
            return True

    @staticmethod
    def check_valid_game_length(info, game_length):
        """Check validity of game length."""
        return info.game_duration_loops > game_length

    @staticmethod
    def get_replay_meta_info(info):
        """Get game replay information."""
        result = {
            "map_name": info.map_name,
            "game_duration_loops": info.game_duration_loops,
            "game_duration_seconds": info.game_duration_seconds,
            "game_version": info.game_version,
            "base_build": info.base_build,
        }
        for info_pb in info.player_info:
            tmp = dict()
            tmp['race_requested'] = sc_common.Race.Name(info_pb.player_info.race_requested)
            tmp['race_actual'] = sc_common.Race.Name(info_pb.player_info.race_actual)
            tmp['result'] = sc_pb.Result.Name(info_pb.player_result.result)
            tmp['apm'] = info_pb.player_apm
            tmp['mmr'] = info_pb.player_mmr
            result[f"Player_{info_pb.player_info.player_id}"] = tmp

        return result

    @staticmethod
    def get_matchup(info):
        """Get match-up string. e.g. TvT, TvP, ..."""
        f2s = {'Terran': 'T', 'Protoss': 'P', 'Zerg': 'Z'}
        races_short = []
        for info_pb in info.player_info:
            race_full = sc_common.Race.Name(info_pb.player_info.race_actual)
            race_short = f2s.get(race_full)
            races_short.append(race_short)
        return 'v'.join(races_short)

    @staticmethod
    def check_valid_matchup(info, matchup):
        """Check if matchup is valid."""
        return ReplayRunner.get_matchup(info) == matchup

    @staticmethod
    def get_map_data(map_dir: str, map_name: str):
        path = os.path.join(map_dir, map_name.replace(" ", "") + ".SC2Map")
        if gfile.Exists(path):
            with gfile.Open(path, "rb") as f:
                return f.read()
        raise ValueError(f"Map {map_name} does not found.")

    def safe_escape(self):
        """Closes client."""
        if self.controller.ping():
            self.controller.quit()


def main(argv):
    """Main function."""

    del argv  # unused

    console = Console()

    # Check flag sanity
    check_flags()

    # Set path to StarCraft II
    os.environ['SC2PATH'] = FLAGS.sc2_path

    def _main(replay_file):
        try:
            runner = ReplayRunner(
                replay_file=replay_file,
                player_id=FLAGS.player_id,
                screen_size=FLAGS.screen_size,
                minimap_size=FLAGS.minimap_size,
                discount=FLAGS.discount,
                step_mul=FLAGS.step_mul,
                override=FLAGS.override,
            )
            parsers = {'SpatialFeatures': SpatialFeatParser}
            runner.start(parsers, console=console)
        except ValueError as e:
            console.print(str(e), style='bold yellow')
            #logging.info(str(e))
        except ConnectionRefusedError as e:
            console.print(str(e), style='bold yellow')
            #logging.info(str(e))
        except UnboundLocalError as e:  # 'agent_obs' is seldomly not assigned
            console.print(str(e), style='bold yellow')
            #logging.info(str(e))
        except KeyboardInterrupt:
            sys.exit()
        finally:
            try:
                runner.safe_escape()
            except UnboundLocalError:
                pass

    if FLAGS.replay_file is not None:
        # Parsing a single replay
        _main(FLAGS.replay_file)
    elif FLAGS.replay_dir is not None:
        # Parsing multiple replays
        replay_files = glob.glob(os.path.join(FLAGS.replay_dir, '**/*.SC2Replay'), recursive=True)
        num_total_replays = len(replay_files)
        for i, replay_file in enumerate(replay_files, 1):
            if i < FLAGS.resume_from:
                continue
            console.print(f"Replay [{i:05d}/{num_total_replays:>05d}] starting: {os.path.split(replay_file)[-1]}", style='bold blue')
            # logging.info(f"Replay file: {os.path.split(replay_file)[-1]}")
            _main(replay_file)
            console.print(f"Replay [{i:>05d}/{num_total_replays:>05d}] terminating.", style='bold red')
            # logging.info(f"Replay [{i+1:>05d}/{num_total_replays:>05d}] terminating.")
            time.sleep(5.)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    app.run(main)
