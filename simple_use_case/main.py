import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from xconfig import xconfig
from model import MyModel

import torch
import os.path as osp


def main():
    
    config = xconfig.xConfig()
    
    config_file = sys.argv[2]
    config.load_yaml(config_file)  # load .yaml config file
    
    cmd_config = sys.argv[3:]
    config.parse(cmd_config)  # parse command line arguments (overwrite the same config in .yaml)
    
    results_root = config['results_root']
    config.save_yaml(osp.join(results_root, 'config.yaml'))  # save as config.yaml
    
    model = MyModel(config, 'model')
    
    opt = torch.optim.Adam(model.param_list())
    
    loss_fn = torch.nn.MSELoss()
    
    def get_loss():
        X = torch.normal(mean=0.0, std=1.0, size=(256, 64))
        X_out = model(X)
        loss = loss_fn(X, X_out)
        return loss
    
    best_loss = 999999
    with torch.no_grad():
        loss = get_loss()
    epoch = 0
    
    while True:
        if not (epoch % 5):
            print(epoch, loss.item())
            if loss.item() > best_loss:
                break
            best_loss = loss.item()
        epoch += 1
        
        opt.zero_grad()
        loss = get_loss()
        loss.backward()
        opt.step()
    

if __name__ == '__main__':
    main()
