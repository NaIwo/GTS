import os
from envyaml import EnvYAML


class ConfigReader:
    def __init__(self):
        pass

    @staticmethod
    def read_config(path: str) -> EnvYAML:
        cfg = EnvYAML(path)
        return cfg


try:
    config = ConfigReader.read_config(os.path.join('..', 'config.yml'))
except FileNotFoundError as e:
    config = ConfigReader.read_config('config.yml')
