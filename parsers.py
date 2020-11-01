# -*- coding: utf-8 -*-

"""
    1. ParserBase
    2. ScreenFeatParser
    3. MinimapFeatParser
    4. CustomSpatialParser
"""

import os
import shutil

import collections
import numpy as np
import scipy.sparse as sp

from absl import flags

FLAGS = flags.FLAGS


class ParserBase(object):
    """Abstract class for replay parsers."""
    def __init__(self):
        pass

    def parse(self, timestep):
        """Must override."""
        raise NotImplementedError

    def save(self):
        """Must override."""
        raise NotImplementedError


class ScreenFeatParser(ParserBase):
    """Parse 'feature_screen' from timestep observation."""

    def __init__(self, sparse: bool = True):
        super(ScreenFeatParser, self).__init__()
        self.screen_features = collections.defaultdict(list)
        self.sparse = sparse
        self.write_file = None

    def parse(self, timestep):
        if self.sparse:
            raise NotImplementedError
        self._append_screen_features(timestep)

    def save(self, path: str):
        if self.sparse:
            raise NotImplementedError
        self._save_screen_features(path)

    def _append_screen_features(self, timestep):
        screen = timestep.observation['feature_screen']
        name2idx = screen._index_names[0]  # pylint: disable=protected-access
        for name, _ in name2idx.items():
            self.screen_features[name].append(screen[name])

    def _save_screen_features(self, path: str):
        assert isinstance(self.screen_features, dict)
        np.savez_compressed(file=path ,**self.screen_features)
        print('{} | Saved screen features to: {}'.format(self.__class__.__name__, path))


class MinimapFeatParser(ParserBase):
    """Parse 'feature_minimap' from timestep observation."""

    def __init__(self, sparse: bool = True):
        super(MinimapFeatParser, self).__init__()
        self.minimap_features = collections.defaultdict(list)
        self.sparse = sparse
        self.write_file = None

    def parse(self, timestep):
        if self.sparse:
            raise NotImplementedError
        self._append_minimap_features(timestep)

    def save(self, path: str):

        if self.sparse:
            raise NotImplementedError
        self._save_minimap_features(path)

    def _append_minimap_features(self, timestep):
        minimap = timestep.observation['feature_minimap']
        name2idx = minimap._index_names[0]  # pylint: disable=protected-access
        for name, _ in name2idx.items():
            self.minimap_features[name].append(minimap[name])

    def _save_minimap_features(self, path: str):
        """Save minimap to .npz format."""
        assert isinstance(self.minimap_features, dict)
        np.savez_compressed(file=path, **self.minimap_features)
        print('{} | Saved minimap features to: {}'.format(self.__class__.__name__, path))


class SpatialFeatParser(ParserBase):
    """
    Parse 'feature_spatial' from timestep observation.
    Note that 'feature_spatial' is a customly implemented feature.
    """
    def __init__(self, sparse: bool = True):
        super(SpatialFeatParser, self).__init__()
        self.spatial_features = collections.defaultdict(list)
        self.sparse = sparse  # bool
        self.write_file = None

    def parse(self, timestep):
        if self.sparse:
            self._append_spatial_features_sparse(timestep)
        else:
            self._append_spatial_features(timestep)

    def save(self, path: str = None, override: bool = False):

        if path is None:
            if self.write_file is None:
                raise AttributeError("File to write has not been specified")
            path = self.write_file

        if not override:
            if os.path.exists(path):
                raise FileExistsError

        if self.sparse:
            self._save_spatial_features_sparse(path)
        else:
            self._save_spatial_features(path)

    def _append_spatial_features(self, timestep):
        """..."""
        spatial = timestep.observation['feature_spatial']
        name2idx = spatial._index_names[0]  # pylint: disable=protected-access
        for name, _ in name2idx.items():
            self.spatial_features[name] += [spatial[name]]

    def _append_spatial_features_sparse(self, timestep):
        """..."""
        spatial = timestep.observation['feature_spatial']
        name2idx = spatial._index_names[0]  # pylint: disable=protected-access
        for name, _ in name2idx.items():
            coo = sp.coo_matrix(spatial[name])
            self.spatial_features[name] += [coo]

    def _save_spatial_features(self, path: str = None):
        """..."""
        assert isinstance(self.spatial_features, dict)
        np.savez_compressed(path,**self.spatial_features)
        print('{} | Saved Spatial features to: {}'.format(self.__class__.__name__, path))

    def _save_spatial_features_sparse(self, path: str):
        """Save 'spatial' to .npz format."""
        assert isinstance(self.spatial_features, dict)

        path = os.path.abspath(path)
        temp_dir  = os.path.join(os.path.dirname(path), 'temp')  # /dir/to/path/temp
        if os.path.isdir(temp_dir):
            raise FileExistsError(f"'{temp_dir}' already exists.")
        os.makedirs(temp_dir, exist_ok=False)

        for name, spatial in self.spatial_features.items():
            for i, spt in enumerate(spatial):
                write_dir = os.path.join(temp_dir, str(name))  # /dir/to/path/temp/name/
                os.makedirs(write_dir, exist_ok=True)
                sp.save_npz(os.path.join(write_dir, f"{i}.npz"), spt)

        self.make_archive(temp_dir, path)
        shutil.rmtree(temp_dir)  # remove temporaries

    @staticmethod
    def make_archive(source: str, destination: str):

        base = os.path.basename(destination)
        name, ext = os.path.splitext(base)  # name.zip -> name, .zip
        ext = ext.strip('.')                # .zip -> zip

        archive_from = os.path.dirname(source)
        archive_to   = os.path.basename(source.strip(os.sep))

        shutil.make_archive(base_name=name, format=ext, root_dir=archive_from, base_dir=archive_to)
        shutil.move(f"{name}.{ext}", destination)
