import json

def load_config(config_file):
    with open(config_file, 'rb') as f:
        config_para = json.load(f)

    return config_para

def save_config(config_file, config_para):
    with open(config_file, 'wb') as f:
        json.dump(config_para, f, indent=4, sort_keys=True)