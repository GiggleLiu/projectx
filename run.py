import yaml
import numpy as np
from controller import *
from multiprocessing import Process

np.random.seed(2)

def load_config(name):
    stream = file(name, 'r')
    yaml_configs = yaml.load(stream)
    print('Project name: %s' % (yaml_configs['project']))
    tasks = yaml_configs['task']
    for task in tasks:
        task[]
