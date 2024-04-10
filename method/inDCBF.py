import torch.nn.functional as F
import torch
import pytorch_lightning as pl
from torch import nn
from torchdiffeq import odeint
from tqdm import trange

def build_mlp(hidden_dims,dropout=0,activation=nn.ReLU,with_bn=True,no_act_last_layer=False):
    modules = []
    for i in range(len(hidden_dims)-1):
        modules.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        if not (no_act_last_layer and i == len(hidden_dims)-2):
            if with_bn:
                modules.append(nn.BatchNorm1d(hidden_dims[i+1]))
            modules.append(activation())
            if dropout > 0.:
                modules.append(nn.Dropout(p=dropout))
    return nn.Sequential(*modules)

class VAE(torch.nn.Module):
    def __init__(self,in_channels,n_control,latent_dim,hidden_dims=None):
        super(VAE, self).__init__()
        encoder = []
        decoder = []
        if hidden_dims is None:
            hidden_dims = [8, 16, 32, 32]
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
        self.encoder_latent = nn.Linear(hidden_dims[-1]*12+latent_dim+n_control, latent_dim)
        hidden_dims.reverse()
        self.decoder_latent = nn.Linear(latent_dim, hidden_dims[0]*12)
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
                            nn.Conv2d(hidden_dims[-1], out_channels= 1,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def forward(self,i,x,u):
        latent = self.encoder(i).view(i.shape[0],-1)
        latent = torch.cat((latent,x,u),1)
        return self.encoder_latent(latent)
    
    def reconstruct(self,latent):
        B,T = latent.shape[:2]
        latent = self.decoder_latent(latent)
        latent = latent.view(B*T,32,3,4)
        latent = self.decoder(latent)
        latent = self.final_layer(latent)
        latent = F.interpolate(latent, size=[36, 64])
        C,H,W = latent.shape[1:]
        return latent.view(B,T,C,H,W)

class NeuralODE(torch.nn.Module):
    def __init__(self,latent_dim,h_dim):
        self.func = build_mlp([latent_dim,h_dim,latent_dim])

class InDCBFController(torch.nn.Module):
    def __init__(self,C,n_control,model_ref,device,latent_dim=256,h_dim=32,gamma=0.5,sample_size=200):
        super(InDCBFController, self).__init__()
        self.model_ref = model_ref
        self.latent_dim = latent_dim
        self.device = device
        self.vae = VAE(C,n_control,latent_dim)
        self.ode_f = build_mlp([latent_dim,h_dim,latent_dim])
        self.ode_g = build_mlp([latent_dim,h_dim,latent_dim*n_control])
        self.n_control = n_control
        self.sample_size = sample_size

    ''' I N*T*H
        u N*T*H
    '''
    def simulate(self,i,u):
        x = torch.rand(i.shape[0],self.latent_dim).to(self.device)
        u = torch.cat([u[:,0,:].unsqueeze(1),u],dim=1)
        xs = []
        x_tides = []
        for k in trange(i.shape[1]):
            def odefunc(t,state):
                g = self.ode_g(state)
                gu = torch.bmm(g.view(g.shape[0],-1,self.n_control),u[:,k+1].unsqueeze(-1))
                return self.ode_f(state) + gu.squeeze(-1)
            timesteps = torch.Tensor([k*0.05,(k+1)*0.05]).to(self.device)
            x_tide = odeint(odefunc,x,timesteps)[1,:,:]
            x = self.vae(i[:,k,:],x,u[:,k])
            xs.append(x)
            x_tides.append(x_tide)
        xs = torch.stack(xs,dim=1)
        x_tides = torch.stack(x_tides,dim=1)
        i_hat = self.vae.reconstruct(xs)
        i_tide = self.vae.reconstruct(x_tides)
        return  (xs,x_tides,i_hat,i_tide)

    def forward(self,i,u,x=None):
        if x is None:
            x = torch.rand(i.shape[0],self.latent_dim)
        x = self.vae(x,i,u)
        for _ in range(self.sample_size):
            us = self.model_ref.generate(x)
            u = self.nac_filter(us)
            if u is not None:
                return u
        return self.model_ref.generate(x)
    
    def loss_function(self,i,i_hat,i_tide,x,x_tide):
        loss_latent = F.mse_loss(x,x_tide)
        loss_dyn = F.mse_loss(i,i_tide)
        loss_recon = F.mse_loss(i,i_hat)
        return {'loss_latent': loss_latent,'loss_dyn': loss_dyn,'loss_recon': loss_recon}

    def save_neural_states(self):
        return

    def nac_filter(self,u):
        return


class InDCBFTrainer(pl.LightningModule):
    def __init__(self,model,args):
        super(InDCBFTrainer,self).__init__()
        self.model = model
        self.args = args
        self.curr_device = None
    
    def forward(self,i,u,x=None):
        return self.model(i,u,x)
    
    def training_step(self, batch, batch_idx):
        i, u = batch
        self.curr_device = i.device

        x,x_tide,i_hat,i_tide = self.model.simulate(i,u)
        train_loss = self.model.loss_function(i,i_hat,i_tide,x,x_tide)
        print(train_loss)
        train_loss['loss'] = train_loss['loss_latent']*self.args['w_latent'] \
               + train_loss['loss_dyn']*self.args['w_dyn'] \
               + train_loss['loss_recon']*self.args['w_recon']
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        i, u = batch
        self.curr_device = i.device

        x,x_tide,i_hat,i_tide = self.model.simulate(i,u)
        val_loss = self.model.loss_function(i,i_hat,i_tide,x,x_tide)
        val_loss['loss'] = val_loss['loss_latent']*self.args['w_latent'] \
               + val_loss['loss_dyn']*self.args['w_dyn'] \
               + val_loss['loss_recon']*self.args['w_recon']
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                               lr=self.args['learning_rate'],
                               weight_decay=self.args['weight_decay'])
        return optimizer