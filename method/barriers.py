import torch
import torch.nn.functional as F
from torchdiffeq import odeint

class iDBFBarrier(torch.nn.Module):
    def __init__(self,
                 n_control,
                 latent_dim,
                 h_dim = 64,
                #  h_dim = 1024,
                 eps_safe = 1,
                 eps_unsafe = 1,
                 eps_ascent = 1,
                 eps_descent = 1,
                 w_safe=1,
                 w_unsafe=1,
                 w_grad=1,
                 w_non_zero=1,
                 w_lambda=1,
                 with_gradient=False,
                 with_nonzero=False,
                 **kwargs
                 ):
        super(iDBFBarrier, self).__init__()
        modules = []
        # hidden_dims = [latent_dim,h_dim,h_dim,h_dim,1]
        hidden_dims = [latent_dim,h_dim,h_dim,1]
        for i in range(len(hidden_dims)-1):
            modules.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            if not i == len(hidden_dims)-2:
                modules.append(torch.nn.ReLU())
        modules.append(torch.nn.Tanh())
        self.cbf = torch.nn.Sequential(*modules)
        self.n_control = n_control
        self.eps_safe = eps_safe
        self.eps_unsafe = eps_unsafe
        self.eps_ascent = eps_ascent
        self.eps_descent = eps_descent
        self.w_safe = w_safe
        self.w_unsafe = w_unsafe
        self.w_grad = w_grad
        self.w_non_zero = w_non_zero
        self.w_lambda = w_lambda
        self.with_gradient = with_gradient
        self.with_nonzero = with_nonzero

    def forward(self,x):
        return self.cbf(x)

    def loss_function(self,x,label,u,ode):
        # x = x.detach()
        label = label.squeeze(dim=-1)
        x_safe = x[(label == 2) + (label == 0)]
        x_unsafe = x[label==1]
        b_safe = self.forward(x_safe)
        b_unsafe = self.forward(x_unsafe)
        eps_safe = self.eps_safe*torch.ones_like(b_safe)
        eps_unsafe = self.eps_unsafe*torch.ones_like(b_unsafe)
        loss_1 = F.relu(eps_safe-b_safe).sum(dim=-1).mean()
        loss_2 = F.relu(eps_unsafe+b_unsafe).sum(dim=-1).mean()
        output = {"loss_safe":self.w_safe*loss_1,"loss_unsafe":self.w_unsafe*loss_2,"b_safe":b_safe.mean(),"b_unsafe":b_unsafe.mean()}
        x_g = x[label == 0]
        # x_g.requires_grad = True
        b = self.forward(x_g)
        d_b_safe = torch.autograd.grad(b.mean(),x_g,retain_graph=True)[0]
        with torch.no_grad():
            f, g = ode(x_g)
        gu = torch.einsum('btha,bta->bth',g.view(g.shape[0],g.shape[1],f.shape[-1],self.n_control),u[label == 0])
        ascent_value = torch.einsum('bth,bth->bt', d_b_safe, (f + gu))
        loss_3 = F.relu(self.eps_ascent - ascent_value.unsqueeze(-1) - 0.5*b).sum(dim=-1).mean()
        output['loss_grad_ascent'] = self.w_grad*loss_3
        output['b_grad_ascent'] = (ascent_value.unsqueeze(-1) + b).mean()
        return output

class SABLASBarrier(torch.nn.Module):
    def __init__(self,
                 n_control,
                 latent_dim,
                 h_dim = 64,
                #  h_dim = 1024,
                 eps_safe = 1,
                 eps_unsafe = 1,
                 eps_ascent = 1,
                 eps_descent = 1,
                 w_safe=1,
                 w_unsafe=1,
                 w_grad=1,
                 w_non_zero=1,
                 w_lambda=1,
                 with_gradient=False,
                 with_nonzero=False,
                 **kwargs
                 ):
        super(SABLASBarrier, self).__init__()
        modules = []
        # hidden_dims = [latent_dim,h_dim,h_dim,h_dim,1]
        hidden_dims = [latent_dim,h_dim,h_dim,1]
        for i in range(len(hidden_dims)-1):
            modules.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            if not i == len(hidden_dims)-2:
                modules.append(torch.nn.ReLU())
        modules.append(torch.nn.Tanh())
        self.cbf = torch.nn.Sequential(*modules)
        self.n_control = n_control
        self.eps_safe = eps_safe
        self.eps_unsafe = eps_unsafe
        self.eps_ascent = eps_ascent
        self.eps_descent = eps_descent
        self.w_safe = w_safe
        self.w_unsafe = w_unsafe
        self.w_grad = w_grad
        self.w_non_zero = w_non_zero
        self.w_lambda = w_lambda
        self.with_gradient = with_gradient
        self.with_nonzero = with_nonzero

    def forward(self,x):
        return self.cbf(x)
    
    def loss_function(self,x,label,u,ode,dynamics):
        label = label.squeeze(dim=-1)
        x_safe = x[(label == 2) + (label == 0)]
        x_unsafe = x[label==1]
        b_safe = self.forward(x_safe)
        b_unsafe = self.forward(x_unsafe)
        loss_1 = F.relu(self.eps_safe-b_safe).sum(dim=-1).mean()
        loss_2 = F.relu(self.eps_unsafe+b_unsafe).sum(dim=-1).mean()
        output = {"loss_safe":self.w_safe*loss_1,"loss_unsafe":self.w_unsafe*loss_2,"b_safe":b_safe.mean(),"b_unsafe":b_unsafe.mean()}
        x_safe = x[label == 0]
        b_safe = self.forward(x_safe)
        x_tide = x_safe[:,1:]
        x_safe = x_safe[:,:-1]
        b_safe = b_safe[:,:-1]
        f,g = ode(x_safe)
        gu = torch.einsum('btha,bta->bth',g.view(g.shape[0],g.shape[1],f.shape[-1],self.n_control),u[label == 0,:-1])
        derive = f + gu
        x_nom = x_safe + derive*0.1
        x_nom = x_nom + (x_tide-x_nom).detach()
        ascent_value = b_safe + (self.forward(x_nom)-b_safe)/0.1
        loss_3 = F.relu(self.eps_ascent-ascent_value).sum(dim=-1).mean()
        output['loss_grad_ascent'] = self.w_grad*loss_3
        output['b_grad_ascent'] = ascent_value.mean()
        return output

class DHBarrier(torch.nn.Module):
    def __init__(self,
                 n_control,
                 latent_dim,
                 h_dim = 1024,
                 gamma_neg = 5,
                 gamma_pos = 1,
                 **kwargs
                 ):
        super(DHBarrier, self).__init__()
        modules = []
        # hidden_dims = [latent_dim,h_dim,h_dim,h_dim,1]
        hidden_dims = [latent_dim,h_dim,h_dim,n_control+1]
        for i in range(len(hidden_dims)-1):
            modules.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            if not i == len(hidden_dims)-2:
                modules.append(torch.nn.ReLU())
        modules.append(torch.nn.Tanh())
        self.cbf = torch.nn.Sequential(*modules)
        self.n_control = n_control
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self,x):
        ab = self.cbf(x)
        if len(x.shape) == 2:
            return ab[:,:-1], ab[:,-1]
        elif len(x.shape) == 3:
            return ab[:,:,:-1], ab[:,:,-1]
    
    def loss_function(self,x,label,u):
        label = label.squeeze(dim=-1)
        x_safe = x[label == 0]
        x_unsafe = x[(label == 1) + (label==2)]
        a_safe, b_safe = self.forward(x_safe)
        a_unsafe, b_unsafe = self.forward(x_unsafe)
        value_safe = torch.einsum("btc,btc->bt",a_safe,u[label == 0]) + b_safe
        value_unsafe = torch.einsum("btc,btc->bt",a_unsafe,u[(label == 1) + (label==2)]) + b_unsafe
        loss_pos = self.gamma_pos*F.relu(value_safe).mean()
        loss_neg = self.gamma_neg*F.relu(-value_unsafe).mean()
        output = {'loss_pos':loss_pos,'loss_neg':loss_neg}
        return output