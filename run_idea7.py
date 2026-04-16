import sys
import yaml
import subprocess
from train import main

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['train']['batch_size'] = 4
config['model']['model_type'] = 'idea7'
config['dataset']['category'] = ['carpet']

with open('config_idea7.yaml', 'w') as f:
    yaml.dump(config, f)

subprocess.run(["python", "train.py", "--config", "config_idea7.yaml"])
