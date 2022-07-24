import yaml


def load_yaml(file):
    with open(file, 'r') as f:
        obj = yaml.load(f, Loader=yaml.FullLoader)
    return obj


def save_yaml(file, obj):
    with open(file, 'w') as f:
        yaml.dump(obj, f, indent=4, sort_keys=False)
