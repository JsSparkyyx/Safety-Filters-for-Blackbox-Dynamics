import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import torch
import os
from torch import nn
from torchdiffeq import odeint
from tqdm import trange
from torchvision.utils import save_image
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

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

class ViTPooler(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class MAE(torch.nn.Module):
    def __init__(self,latent_dim,n_control=2,model="google/vit-base-patch16-224",vit_dim=768,num_cam=6,freeze_ViT=True):
        super(MAE, self).__init__()
        if "mae" in model:
            from transformers import ViTMAEForPreTraining
            self.model = ViTMAEForPreTraining.from_pretrained(model)
            self.encoder = self.model.vit
            self.decoder = self.model.decoder
        if freeze_ViT:
            for n,p in self.model.named_parameters():
                if "pooler" not in n:
                    p.requires_grad = False
        self.mlp = build_mlp([vit_dim+num_cam,latent_dim,latent_dim])
        self.attention = torch.nn.parameter.Parameter(torch.rand((num_cam,latent_dim)))
        self.linear = torch.nn.Linear(2*latent_dim+n_control,latent_dim)
        self.num_cam = num_cam
    
    def forward(self,imgs,x_p,u_p):
        B,N,C,H,W = imgs.shape
        with torch.no_grad():
            outputs = self.encoder(pixel_values=imgs.reshape(-1,C,H,W))
        pos_encoding = torch.eye(self.num_cam).expand(imgs.shape[0],-1,-1).to(imgs.device)
        rep = torch.cat([outputs.last_hidden_state.mean(1).reshape(B,N,-1),pos_encoding],dim=-1)
        rep = self.mlp(rep)
        weight = torch.einsum("bch,bch->bc",self.attention.expand(B,-1,-1),rep)
        final_rep = torch.einsum("bn,bnh->bnh",weight,rep).sum(1)
        final_rep = torch.cat([final_rep,x_p,u_p],dim=-1)
        return self.linear(final_rep)

class InDCBFController(torch.nn.Module):
    def __init__(self,n_control,device,model,latent_dim=256,h_dim=1024):
        super(InDCBFController, self).__init__()
        self.latent_dim = latent_dim
        self.device = device
        self.encoder = MAE(latent_dim,model=model)
        self.ode = NeuralODE([latent_dim,h_dim,h_dim,latent_dim],
                             [latent_dim,h_dim,h_dim,latent_dim*n_control])
        self.n_control = n_control

    def forward(self,i,u_p,x_p,u_ref,barrier):
        u_ref = u_ref.view(-1).cpu().numpy()
        x = self.encoder(i,x_p,u_p)
        f, g = self.ode(x)
        f = f.view(-1).detach().cpu().numpy()
        g = g.view(-1,2).detach().cpu().numpy()
        b = barrier(x)
        d_b = torch.autograd.grad(b,x,retain_graph=True)[0]
        b = b.view(-1).detach().cpu().numpy()
        x = x.view(-1).detach().cpu().numpy()
        d_b = d_b.view(-1).cpu().numpy()
        u = cp.Variable(u_ref.shape)
        t1 = d_b @ f
        t2 = d_b @ g 
        t3 = b
        objective = cp.Minimize(cp.sum_squares(u - u_ref))
        constraints = [(t1+t2@u+t3)>=0]
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        return u, result, prob

    def simulate(self,i,u,dt=0.1,window_size=5,rtol=5e-6):
        x_init = torch.zeros(i.shape[0],self.latent_dim).to(self.device)
        u = torch.cat([u[:,0,:].unsqueeze(1),u],dim=1)
        x = self.encoder(i[:,0,:],x_init,u[:,0])
        x_tide = x
        xs = [x]
        x_tides = [x_tide]
        for k in trange(1,i.shape[1]):
            if k % window_size == 1:
                x_tide = x
            def odefunc(t,state):
                f, g = self.ode(state)
                gu = torch.bmm(g.view(g.shape[0],-1,self.n_control),u[:,k+1].unsqueeze(-1))
                return f + gu.squeeze(-1)
            timesteps = torch.Tensor([0,dt]).to(self.device)
            x_tide = odeint(odefunc,x_tide,timesteps,rtol=rtol)[1,:,:]
            x = self.encoder(i[:,k,:],x,u[:,k])
            xs.append(x)
            x_tides.append(x_tide)
        xs = torch.stack(xs,dim=1)
        x_tides = torch.stack(x_tides,dim=1)
        i_hat = self.vae.reconstruct(xs)
        i_tide = self.vae.reconstruct(x_tides)
        return (xs,x_tides,i_hat,i_tide)
    
    def loss_function(self,i,i_hat,i_tide,x,x_tide):
        loss_latent = F.mse_loss(x,x_tide)
        loss_dyn = F.mse_loss(i,i_tide)
        loss_recon = F.mse_loss(i,i_hat)
        return {'loss_latent': loss_latent,'loss_dyn': loss_dyn,'loss_recon': loss_recon}

class NeuralODE(nn.Module):
    def __init__(self,params_f,params_g):
        super(NeuralODE, self).__init__()
        self.ode_f = build_mlp(params_f)
        self.ode_g = build_mlp(params_g)
        self.num_f = len(params_f)-1
        self.num_g = len(params_g)-1

    def forward(self,x):
        return self.ode_f(x),self.ode_g(x)
    
class Barrier(torch.nn.Module):
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
        super(Barrier, self).__init__()
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
        x_safe = x[label == 0]
        x_unsafe = x[label == 1]
        b_safe = self.forward(x_safe)
        b_unsafe = self.forward(x_unsafe)
        eps_safe = self.eps_safe*torch.ones_like(b_safe)
        eps_unsafe = self.eps_unsafe*torch.ones_like(b_unsafe)
        loss_1 = F.relu(eps_safe-b_safe).sum(dim=-1).mean()
        loss_2 = F.relu(eps_unsafe+b_unsafe).sum(dim=-1).mean()
        output = {"loss_safe":self.w_safe*loss_1,"loss_unsafe":self.w_unsafe*loss_2,"b_safe":b_safe.mean(),"b_unsafe":b_unsafe.mean()}
        x_g = x_safe.clone().detach()
        x_g.requires_grad = True
        b = self.forward(x_g)
        d_b_safe = torch.autograd.grad(b.mean(),x_g,retain_graph=True)[0]
        with torch.no_grad():
            f, g = ode(x_g)
        gu = torch.einsum('btha,bta->bth',g.view(g.shape[0],g.shape[1],f.shape[-1],self.n_control),u[label == 0])
        ascent_value = torch.einsum('bth,bth->bt', d_b_safe, (f + gu))
        loss_3 = F.relu(self.eps_ascent - ascent_value.unsqueeze(-1) - b_safe).sum(dim=-1).mean()
        output['loss_grad_ascent'] = self.w_grad*loss_3
        output['b_grad_ascent'] = (ascent_value.unsqueeze(-1) + b_safe).mean()
        return output
        
class InDCBFTrainer(pl.LightningModule):
    def __init__(self,
                 model,
                 barrier = None,
                 learning_rate=0.001,
                 weight_decay=0,
                 w_barrier=2,
                 w_latent=1,
                 w_dyn=1,
                 w_recon=1,
                 window_size=5,
                 rtol=5e-6,
                 dt=0.05,
                 with_dynamic=True,
                 train_barrier=False,
                 **kwargs):
        super(InDCBFTrainer,self).__init__()
        self.model = model
        self.barrier = barrier
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.window_size = window_size
        self.rtol = rtol
        self.dt = dt
        self.w_latent = w_latent
        self.w_barrier = w_barrier
        self.w_dyn = w_dyn
        self.w_recon = w_recon
        self.curr_device = None
        self.train_barrier = train_barrier
        self.with_dynamic = with_dynamic
        self.save_hyperparameters(ignore=['model','barrier'])
        # print('----hyper parameters----')
        # print(self.hparams)
    
    def forward(self,i,u,x=None):
        return self.model(i,u,x)
    
    def training_step(self, batch, batch_idx):
        i, u, label = batch
        self.curr_device = i.device

        x,x_tide,i_hat,i_tide = self.model.simulate(i,u,dt=self.dt,window_size=self.window_size,rtol=self.rtol)
        train_loss = self.model.loss_function(x,x_tide)
        train_loss['loss'] = 0
        if self.with_dynamic:
            train_loss['loss'] += train_loss['loss_latent']*self.w_latent
            train_loss['loss'] += train_loss['loss_dyn']*self.w_latent
            train_loss['loss'] += train_loss['loss_recon']*self.w_latent
        if self.train_barrier:
            output = self.barrier.loss_function(x,label,u,self.model.ode)
            train_loss['loss_safe'] = output['loss_safe']
            train_loss['loss_unsafe'] = output['loss_unsafe']
            train_loss['loss_grad_ascent'] = output['loss_grad_ascent']
            train_loss['loss'] += output['loss_safe']*self.w_barrier+output['loss_unsafe']*self.w_barrier
            train_loss['loss'] += output['loss_grad_ascent']*self.w_barrier
            self.log_dict({'b_safe':output['b_safe'],'b_unsafe':output['b_unsafe']},sync_dist=True)
            self.log_dict({'b_grad_ascent':output['b_grad_ascent']},sync_dist=True)
            if batch_idx % 5 == 0:
                print()
                print(output['b_safe'])
                print(output['b_unsafe'])
                print(output['b_grad_ascent'])
                print()
                print(train_loss)
                print()
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        i, u, label = batch
        self.curr_device = i.device

        x,x_tide = self.model.simulate(i,u,dt=self.dt,window_size=self.window_size,rtol=self.rtol)
        val_loss = self.model.loss_function(x,x_tide)
        val_loss['loss'] = 0
        if self.with_dynamic:
            val_loss['loss'] += val_loss['loss_latent']*self.w_latent
            val_loss['loss'] += val_loss['loss_dyn']*self.w_latent
            val_loss['loss'] += val_loss['loss_recon']*self.w_latent
        if self.train_barrier:
            output = self.barrier.loss_function(x,label,u,self.model.ode)
            val_loss['loss_safe'] = output['loss_safe']
            val_loss['loss_unsafe'] = output['loss_unsafe']
            val_loss['loss_grad_ascent'] = output['loss_grad_ascent']
            val_loss['loss'] += output['loss_safe']*self.w_barrier+output['loss_unsafe']*self.w_barrier
            val_loss['loss'] += output['loss_grad_ascent']*self.w_barrier
            self.log_dict({'val_b_safe':output['b_safe'],'val_b_unsafe':output['b_unsafe']},sync_dist=True)
            self.log_dict({'val_b_grad_ascent':output['b_grad_ascent']},sync_dist=True)
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
    
    def on_train_epoch_end(self) -> None:
        # min_eps = 0.1
        # step_eps = (0.25-min_eps)/100
        # max_w = 1
        # step_w = (max_w-0.4)/100
        # self.barrier.eps_unsafe -= step_eps
        # self.barrier.eps_ascent -= step_eps
        # self.barrier.w_unsafe += step_w
        pass

    def on_validation_end(self) -> None:
        # self.sample_states()
        pass
    
    # def on_train_epoch_start(self):
    #     if self.current_epoch == 10:
    #         self.barrier.w_safe = 5
    #         self.barrier.w_unsafe = 1
    #         self.barrier.w_grad = 1

    def sample_states(self):          
        i, u, label = next(iter(self.trainer.datamodule.val_dataloader()))
        i = i.to(self.curr_device)
        u = u.to(self.curr_device)

        x,x_tide = self.model.simulate(i,u)
        np.savetxt(os.path.join(self.logger.log_dir , 
                                       "Latent", 
                                       f"latent_Epoch_{self.current_epoch}.txt"),
                                       x.data[0].cpu().numpy())
        np.savetxt(os.path.join(self.logger.log_dir , 
                                       "LatentDynamic", 
                                       f"latent_dynamic_Epoch_{self.current_epoch}.txt"),
                                       x_tide.data[0].cpu().numpy())
        
    def configure_optimizers(self):
        params = [{"params":self.model.parameters(),"lr":self.learning_rate,"weight_decay":self.learning_rate}]
        if self.train_barrier:
            params.append({"params":self.barrier.parameters(),"lr":self.learning_rate,"weight_decay":self.learning_rate}
                                    )
        optimizer = torch.optim.Adam(params)
        return optimizer
    
if __name__ == "__main__":
    model = Encoder(512)
    print(model)