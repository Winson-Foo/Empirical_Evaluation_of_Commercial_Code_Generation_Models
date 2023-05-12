import json

def read_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config