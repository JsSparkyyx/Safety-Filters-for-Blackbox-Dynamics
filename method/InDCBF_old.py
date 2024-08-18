import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import torch
import os
from torch import nn
from torchdiffeq import odeint
from tqdm import trange, tqdm
from torchvision.utils import save_image

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

class VAE(torch.nn.Module):
    def __init__(self,in_channels,n_control,latent_dim,hidden_dims=None):
        super(VAE, self).__init__()
        encoder = []
        decoder = []
        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 128, 256, 512]
        for h_dim in hidden_dims:
            encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                                kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*encoder)
        self.encoder_latent = nn.Linear(hidden_dims[-1]*16+latent_dim+n_control, latent_dim)
        hidden_dims.reverse()
        self.decoder_latent = nn.Linear(latent_dim, hidden_dims[0]*16)
        for i in range(len(hidden_dims) - 1):
            decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                        hidden_dims[i + 1],
                                        kernel_size=3,
                                        stride = 2,
                                        padding=1,
                                        output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
        )
        self.decoder = nn.Sequential(*decoder)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())
        self.hidden_dims = hidden_dims

    def forward(self,i,x,u):
        latent = self.encoder(i).view(i.shape[0],-1)
        latent = torch.cat((latent,x,u),1)
        return self.encoder_latent(latent)
    
    def reconstruct(self,latent):
        B,T = latent.shape[:2]
        latent = self.decoder_latent(latent)
        latent = latent.view(B*T,self.hidden_dims[0],4,4)
        latent = self.decoder(latent)
        latent = self.final_layer(latent)
        latent = F.interpolate(latent, size=[224, 224])
        C,H,W = latent.shape[1:]
        return latent.view(B,T,C,H,W)

class NeuralODE(nn.Module):
    def __init__(self,params_f,params_g):
        super(NeuralODE, self).__init__()
        self.ode_f = build_mlp(params_f,with_bn=False)
        self.ode_g = build_mlp(params_g,with_bn=False)
        self.num_f = len(params_f)-1
        self.num_g = len(params_g)-1

    def forward(self,x):
        f = x
        g = x
        for layer in self.ode_f.keys():
            f = self.ode_f[layer](f)
        for layer in self.ode_g.keys():
            g = self.ode_g[layer](g)
        return f,g

class InDCBFController(torch.nn.Module):
    def __init__(self,C,n_control,model_ref,device,latent_dim=256,h_dim=256):
        super(InDCBFController, self).__init__()
        self.model_ref = model_ref
        self.latent_dim = latent_dim
        self.device = device
        self.vae = VAE(C,n_control,latent_dim)
        self.ode = NeuralODE([latent_dim,h_dim,h_dim,latent_dim],
                             [latent_dim,h_dim,h_dim,latent_dim*n_control])
        self.n_control = n_control

    def simulate(self,i,u,dt=0.1,window_size=5,rtol=5e-6):
        x_init = torch.zeros(i.shape[0],self.latent_dim).to(self.device)
        u = torch.cat([u[:,0,:].unsqueeze(1),u],dim=1)
        x = self.vae(i[:,0,:],x_init,u[:,0])
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
            x = self.vae(i[:,k,:],x,u[:,k])
            xs.append(x)
            x_tides.append(x_tide)
        xs = torch.stack(xs,dim=1)
        x_tides = torch.stack(x_tides,dim=1)
        i_hat = self.vae.reconstruct(xs)
        i_tide = self.vae.reconstruct(x_tides)
        return  (xs,x_tides,i_hat,i_tide)

    def forward(self,i,u,x=None,dt=0.05,threshold=0.5,sample_size=200,test_size=20):
        if x is None:
            x = torch.zeros(i.shape[0],self.latent_dim)
        x = self.vae(x,i,u)
        for _ in range(sample_size):
            u = self.model_ref.generate(x)
            scores = self.nac_filter(u,x,dt,test_size)
            if scores >= threshold:
                return u
        return self.model_ref.generate(x)
    
    def loss_function(self,i,i_hat,i_tide,x,x_tide):
        loss_latent = F.mse_loss(x,x_tide)
        loss_dyn = F.mse_loss(i,i_tide)
        loss_recon = F.mse_loss(i,i_hat)
        return {'loss_latent': loss_latent,'loss_dyn': loss_dyn,'loss_recon': loss_recon}

class Barrier(torch.nn.Module):
    def __init__(self,
                 latent_dim,
                 n_control,
                 h_dim = 512,
                 eps_safe = 0.001,
                 eps_unsafe = 0.001,
                 eps_ascent = 0.001,
                 w_safe = 0.5,
                 w_unsafe = 0.5,
                 w_ascent = 0.3,
                 ):
        super(Barrier, self).__init__()
        self.cbf = build_mlp([latent_dim,h_dim,h_dim,1],with_bn=False)
        self.eps_safe = eps_safe
        self.n_control = n_control
        self.eps_unsafe = eps_unsafe
        self.eps_ascent = eps_ascent
        self.w_safe = w_safe
        self.w_unsafe = w_unsafe
        self.w_ascent = w_ascent

    def forward(self,x):
        for layer in self.cbf.keys():
            x = self.cbf[layer](x)
        return x

    def loss_function(self,x,label,u,ode,dt = 0.1,rtol=5e-6):
        N = label.shape[0]
        label = label.squeeze(dim=-1)
        N_unsafe = label.sum()
        N_safe = N - N_unsafe
        x_safe = x[label == 0]
        x_unsafe = x[label == 1]
        b_safe = self.forward(x_safe)
        b_unsafe = self.forward(x_unsafe)
        loss_1 = 1*F.relu(self.eps_safe-b_safe).sum()/(1e-5 + N_safe)
        loss_2 = 2*F.relu(self.eps_unsafe+b_unsafe).sum()/(1e-5 + N_unsafe)
        x_g = x_safe.clone().detach()
        x_g.requires_grad = True
        b = self.forward(x_g)
        d_b_safe = torch.autograd.grad(b.mean(),x_g,retain_graph=True)[0]
        with torch.no_grad():
            f, g = ode(x_safe)
        print(b_safe.flatten())
        gu = torch.einsum('btha,bta->bth',g.view(g.shape[0],g.shape[1],-1,self.n_control),u[label == 0])
        ascent_value = torch.einsum('bth,bth->bt', d_b_safe, (f + gu))
        print(d_b_safe)
        loss_3 = 1*F.relu(self.eps_ascent - ascent_value.unsqueeze(-1) + b_safe).sum()/(1e-5 + N_safe)
        return loss_1, loss_2, loss_3
    
class InDCBFTrainer(pl.LightningModule):
    def __init__(self,model,barrier,learning_rate=0.001,weight_decay=0,w_barrier=10,w_latent=5,w_dyn=5,w_recon=0.5,window_size=5,rtol=5e-6,dt=0.05,**kwargs):
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
        self.save_hyperparameters(ignore=['model'])
        print('----hyper parameters----')
        print(self.hparams)
    
    def forward(self,i,u,x=None):
        return self.model(i,u,x)
    
    def training_step(self, batch, batch_idx):
        i, u, label = batch
        self.curr_device = i.device

        x,x_tide,i_hat,i_tide = self.model.simulate(i,u,dt=self.dt,window_size=self.window_size,rtol=self.rtol)
        train_loss = self.model.loss_function(i,i_hat,i_tide,x,x_tide)
        # barrier_loss = self.barrier.loss_function(x,label,u,self.model.ode,dt=self.dt,rtol=self.rtol)
        # train_loss['barrier_loss0'] = barrier_loss[0]
        # train_loss['barrier_loss1'] = barrier_loss[1]
        # train_loss['barrier_loss2'] = barrier_loss[2]
        train_loss['loss'] = train_loss['loss_latent']*self.w_latent \
               + train_loss['loss_dyn']*self.w_dyn \
               + train_loss['loss_recon']*self.w_recon \
            #    + (train_loss['barrier_loss0'])*self.w_barrier\
            #    + (train_loss['barrier_loss1'])*self.w_barrier\
            #    + (train_loss['barrier_loss2'])*self.w_barrier
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        # print()
        # print()
        # print(train_loss['barrier_loss0'])
        # print(train_loss['barrier_loss1'])
        # print(train_loss['barrier_loss2'])
        # print(train_loss['loss_latent'])
        # print(train_loss['loss_dyn'])
        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        i, u, label = batch
        self.curr_device = i.device

        x,x_tide,i_hat,i_tide = self.model.simulate(i,u,dt=self.dt,window_size=self.window_size,rtol=self.rtol)
        val_loss = self.model.loss_function(i,i_hat,i_tide,x,x_tide)
        # barrier_loss = self.barrier.loss_function(x,label,u,self.model.ode,dt=self.dt,rtol=self.rtol)
        # val_loss['barrier_loss'] = barrier_loss
        val_loss['loss'] = val_loss['loss_latent']*self.w_latent \
               + val_loss['loss_dyn']*self.w_dyn \
               + val_loss['loss_recon']*self.w_recon \
            #    + val_loss['barrier_loss']*self.w_barrier
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        self.sample_images()

    def sample_images(self):          
        i, u, label = next(iter(self.trainer.datamodule.test_dataloader()))
        i = i.to(self.curr_device)
        u = u.to(self.curr_device)

        x,x_tide,i_hat,i_tide = self.model.simulate(i,u)
        save_image(i_hat.data[0],
                          os.path.join(self.logger.log_dir , 
                                       "ReconDecode", 
                                       f"recon_decode_Epoch_{self.current_epoch}.png"),
                              nrow=self.window_size)
        save_image(i_tide.data[0],
                          os.path.join(self.logger.log_dir , 
                                       "ReconDynamic", 
                                       f"recon_dynamic_Epoch_{self.current_epoch}.png"),
                              nrow=self.window_size)
        save_image(i.data[0],
                          os.path.join(self.logger.log_dir , 
                                       "Samples", 
                                       f"sample_Epoch_{self.current_epoch}.png"),
                              nrow=self.window_size)
        np.savetxt(os.path.join(self.logger.log_dir , 
                                       "Latent", 
                                       f"latent_Epoch_{self.current_epoch}.txt"),
                                       x.data[0].cpu().numpy())
        np.savetxt(os.path.join(self.logger.log_dir , 
                                       "LatentDynamic", 
                                       f"latent_dynamic_Epoch_{self.current_epoch}.txt"),
                                       x_tide.data[0].cpu().numpy())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{"params":self.model.parameters(),"lr":self.learning_rate,"weight_decay":self.learning_rate},
                                    #   {"params":self.barrier.parameters(),"lr":self.learning_rate,"weight_decay":self.learning_rate}
                                    ],)
        return optimizer