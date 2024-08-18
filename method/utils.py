import torch

def build_mlp(hidden_dims,dropout=0,activation=torch.nn.ReLU,with_bn=False,no_act_last_layer=False):
    modules = []
    for i in range(len(hidden_dims)-1):
        modules.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        if not (no_act_last_layer and i == len(hidden_dims)-2):
            if with_bn:
                modules.append(torch.nn.BatchNorm1d(hidden_dims[i+1]))
            modules.append(activation())
            if dropout > 0.:
                modules.append(torch.nn.Dropout(p=dropout))
    return torch.nn.Sequential(*modules)

class NeuralODE(torch.nn.Module):
    def __init__(self,params_f,params_g):
        super(NeuralODE, self).__init__()
        self.ode_f = build_mlp(params_f)
        self.ode_g = build_mlp(params_g)
        self.num_f = len(params_f)-1
        self.num_g = len(params_g)-1

    def forward(self,x):
        return self.ode_f(x),self.ode_g(x)
    
class BlackBoxNODE(torch.nn.Module):
    def __init__(self,params):
        super(BlackBoxNODE, self).__init__()
        self.ode = build_mlp(params)
        self.num = len(params)-1

    def forward(self,x,u):
        x = torch.cat([x,u],dim=-1)
        return self.ode(x)