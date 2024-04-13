import torch.nn as nn
import torch

def build_mlp(hidden_dims,dropout=0,activation=nn.ReLU,with_bn=True,no_act_last_layer=False):
    modules = nn.ModuleDict()
    for i in range(len(hidden_dims)-1):
        modules[f'linear_{i}'] = nn.Linear(hidden_dims[i], hidden_dims[i+1])
        if not (no_act_last_layer and i == len(hidden_dims)-2):
            if with_bn:
                modules[f'batchnorm_{i}'] = nn.BatchNorm1d(hidden_dims[i+1])
            modules[f'activation_{i}'] = activation()
            if dropout > 0.:
                modules[f'dropout_{i}'] = nn.Dropout(p=dropout)
    return modules

class ODE(nn.Module):
    def __init__(self):
        super(ODE, self).__init__()
        self.ode_f = build_mlp([128,128,128,256])
        self.ode_g = build_mlp([128,128,128,256*2])

    def forward(self,x):
        f = x
        g = x
        for layer in self.ode_f.keys():
            f = self.ode_f[layer](f)
        for layer in self.ode_g.keys():
            g = self.ode_f[layer](g)
        return f,g

    def get_layer_output(self,x):
        f = x
        g = x
        layer_state_dict = {}
        for layer in self.ode_f.keys():
            f = self.ode_f[layer](f)
            if 'linear' in layer:
                layer_state_dict[f'f_{layer}'] = f
        for layer in self.ode_g.keys():
            g = self.ode_f[layer](g)
            if 'linear' in layer:
                layer_state_dict[f'g_{layer}'] = g
        return layer_state_dict

L = ODE()
x = torch.rand(16,128)
f,g = L(x)
print(f.shape,g.shape)
print(L.get_layer_output(x).keys())