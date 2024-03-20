import sys

from omegaconf import OmegaConf

from skeleton_tools.skeleton_visualization.data_prepare.data_extract import *
from skeleton_tools.skeleton_visualization.paint_components.dynamic_graphs.dynamic_graphs import *
from skeleton_tools.skeleton_visualization.paint_components.frame_painters.base_painters import *
from skeleton_tools.skeleton_visualization.paint_components.frame_painters.local_painters import *
from skeleton_tools.utils.tools import load_config


def class_of(classname):
    return getattr(sys.modules[__name__], classname)


def replace(cfg, placeholder, value):
    for k, v in cfg.items():
        if isinstance(v, dict):
            replace(v, placeholder, value)
        elif type(v) == str and v == placeholder:
            cfg[k] = value


class ComponentsLoader:
    def __init__(self, cfg):
        self.config = cfg
        self.config['name'], self.config['ext'] = osp.splitext(osp.basename(cfg['paths']['video']))
        self.set_param('GRAPH_LAYOUT', class_of(self.config['$GRAPH_LAYOUT']))

    def _create_local_painter(self, name, **kwargs):
        return self.__create_painter('local', name, **kwargs)

    def _create_global_painter(self, name, **kwargs):
        return self.__create_painter('global', name, **kwargs)

    def __create_painter(self, type, name, **kwargs):
        args = dict(self.config['painters'][type][name]) | kwargs
        return class_of(name)(**args)

    def _create_graph(self, name, **kwargs):
        width, height = self.config['$DATA']['resolution']
        width, height = int(width * self.config['graphs'][name]['width']), int(height * self.config['graphs'][name]['height'])
        args = self.config['graphs'][name] | {'width': width, 'height': height} | kwargs
        return class_of(name)(**args)

    def extract_data(self, **kwargs):
        cfg = self.config['model']['extractor']
        name, args = cfg['name'], cfg['args']
        extractor = class_of(name)(**args | kwargs)
        data = extractor(self.config)
        self.set_param('DATA', data)
        self.set_param('SIGNAL', data['predictions'])
        return data

    def create_painters(self, use_globals=True):
        local_painters = [self._create_local_painter(p) for p in self.config['painters']['local']]
        global_painters = [self._create_global_painter(p) for p in self.config['painters']['global']]
        if use_globals:
            global_painters += [GlobalPainter(p) for p in local_painters]
            local_painters = []
        return local_painters, global_painters

    def create_graphs(self):
        return [self._create_graph(p) for p in self.config['graphs']]

    def set_param(self, name, value):
        self.config[f'${name}'] = value
        replace(self.config, f'${name}', value)

