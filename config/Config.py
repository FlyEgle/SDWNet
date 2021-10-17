import yaml
from dotmap import DotMap

class Config(object):
    def __init__(self, config_file):
        super(Config, self).__init__()
        self.config_file = config_file

    def __call__(self):
        with open(self.config_file, 'r') as file:
            self.yaml_data = yaml.load(file)
        self.cfg = DotMap(self.yaml_data)
        return self.cfg