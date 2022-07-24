from xconfig import xconfig

import torch


class FeedforwardNet(torch.nn.Module):
    
    def __init__(self, global_config, config_field):
        super(FeedforwardNet, self).__init__()
        self.global_config = global_config
        self.config_field = config_field
        self.config = global_config[config_field]
        
        arch = self.config['arch']
        if isinstance(arch, list):
            arch = list(map(eval, arch))
        else:
            arch = eval(arch)
        self.mlp = torch.nn.Sequential(*arch)
        
        self._param_list = [
            {'params': self.mlp.parameters(), 'lr': self.config['lr']}
        ]
        
    def forward(self, x):
        return self.mlp(x)
    
    def param_list(self):
        return self._param_list


class ScaleNet(torch.nn.Module):
    
    def __init__(self, global_config, config_field):
        super(ScaleNet, self).__init__()
        self.global_config = global_config
        self.config_field = config_field
        self.config = global_config[config_field]
        
        arch = self.config['arch']
        if isinstance(arch, list):
            arch = list(map(eval, arch))
        else:
            arch = eval(arch)
        self.mlp = torch.nn.Sequential(*arch)
        
        self._param_list = [
            {'params': self.mlp.parameters(), 'lr': self.config['lr']}
        ]
        
    def param_list(self):
        return self._param_list
    
    def forward(self, x):
        return self.mlp(x)


class MyModel(torch.nn.Module):
    
    def __init__(self, global_config, config_field):
        super(MyModel, self).__init__()
        self.global_config = global_config
        self.config_field = config_field
        self.config = global_config[config_field]
        
        self.fead_forward_net = FeedforwardNet(
            self.global_config, xconfig.join(self.config_field, 'fead_forward_net')
        )
        self.scale_net = ScaleNet(
            self.global_config, xconfig.join(self.config_field, 'scale_net')
        )
        
        self._param_list = []
        self._param_list.extend(self.fead_forward_net.param_list())
        self._param_list.extend(self.scale_net.param_list())
    
    def param_list(self):
        return self._param_list
    
    def forward(self, x):
        return self.scale_net(x) * self.fead_forward_net(x)
