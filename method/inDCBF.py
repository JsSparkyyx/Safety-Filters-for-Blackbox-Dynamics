import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import torch
import os
from torch import nn
from torchdiffeq import odeint
from tqdm import trange, tqdm
from torchvision.utils import save_image
from .utils import sigmoid, Estimator

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
            hidden_dims = [8, 16, 32, 64]
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
        self.hidden_dims = hidden_dims

    def forward(self,i,x,u):
        latent = self.encoder(i).view(i.shape[0],-1)
        latent = torch.cat((latent,x,u),1)
        return self.encoder_latent(latent)
    
    def reconstruct(self,latent):
        B,T = latent.shape[:2]
        latent = self.decoder_latent(latent)
        latent = latent.view(B*T,self.hidden_dims[0],3,4)
        latent = self.decoder(latent)
        latent = self.final_layer(latent)
        latent = F.interpolate(latent, size=[36, 64])
        C,H,W = latent.shape[1:]
        return latent.view(B,T,C,H,W)

class NeuralODE(nn.Module):
    def __init__(self,params_f,params_g):
        super(NeuralODE, self).__init__()
        self.ode_f = build_mlp(params_f)
        self.ode_g = build_mlp(params_g)
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

    def get_layer_output(self,x):
        f = x
        g = x
        layer_output_dict = {}
        for layer in self.ode_f.keys():
            f = self.ode_f[layer](f)
            if 'linear' in layer and str(self.num_f-1) not in layer:
                layer_output_dict[f'f_{layer}'] = [f]
        for layer in self.ode_g.keys():
            g = self.ode_g[layer](g)
            if 'linear' in layer and str(self.num_g-1) not in layer:
                layer_output_dict[f'g_{layer}'] = [g]
        for layer in layer_output_dict.keys():
            if layer.startswith('f_'):
                layer_output_dict[layer].append(f)
            else:
                layer_output_dict[layer].append(g)
        return layer_output_dict

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

    def simulate(self,i,u,dt=0.05,window_size=5,rtol=5e-6):
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

    def init_nac_estimator(self,params):
        self.estimator = {}
        self.params = params
        x = torch.rand(2,self.latent_dim).to(self.device)
        layer_state_dict = self.ode.get_layer_output(x)
        for n, state in layer_state_dict.items():
            self.estimator[n] = Estimator(state.shape[1], params[n]['M'], params[n]['O'], self.device)
        self.layer_num = len(layer_state_dict)

    def save_neural_states(self,data_loader):
        self.eval()
        print('Accessing neural states')
        for i, u in tqdm(data_loader):
            x_init = torch.zeros(i.shape[0],self.latent_dim).to(self.device)
            u = torch.cat([u[:,0,:].unsqueeze(1),u],dim=1)
            x = self.vae(i[:,0,:],x_init,u[:,0])
            xs = []
            for k in trange(1,i.shape[1]):
                x = self.vae(i[:,k,:],x,u[:,k])
                xs.append(x)
            xs = torch.cat(xs,0)
            layer_output_dict = self.ode.get_layer_output(xs)
            for idx, layer_name, output in enumerate(layer_output_dict.items()):
                retain_graph = False if idx == self.layer_num - 1 else True
                states = self.compute_states(output[0], output[1], self.params[layer_name]['sig_alpha'], retain_graph=retain_graph)
                if len(states) > 0:
                    self.estimator_dict[layer_name].update(states)
        for layer_name, output in layer_output_dict.items():
            self.estimator_dict[layer_name].get_score()

    def compute_states(self, h, outputs, sig_alpha, retain_graph=False, **kwargs):
        baseline = torch.zeros_like(outputs)
        loss = F.mse_loss(baseline,outputs)
        layer_grad = torch.autograd.grad(loss.sum(), h, create_graph=False,
                                        retain_graph=retain_graph, **kwargs)[0]
        states = sigmoid(h*layer_grad, sig_alpha=sig_alpha)
        return states

    def nac_filter(self,u,x,dt,test_size):
        def odefunc(t,state):
            f, g = self.ode(state)
            gu = torch.bmm(g.view(g.shape[0],-1,self.n_control),u.unsqueeze(-1))
            return f + gu.squeeze(-1)
        timesteps = torch.Tensor([0,dt]).to(self.device)
        x_tide = odeint(odefunc,x,timesteps,rtol=5e-6)[1,:,:]
        scores = torch.zeros(x_tide.shape[0]).to(self.device)
        for _ in range(test_size):
            u_p = self.model_ref.generate(x_tide)
            def odefunc(t,state):
                f, g = self.ode(state)
                gu = torch.bmm(g.view(g.shape[0],-1,self.n_control),u_p.unsqueeze(-1))
                return f + gu.squeeze(-1)
            timesteps = torch.Tensor([0,dt]).to(self.device)
            x_tide = odeint(odefunc,x_tide,timesteps,rtol=5e-6)[1,:,:]
            layer_output_dict = self.ode.get_layer_output(x_tide)
            for idx, layer_name, output in enumerate(layer_output_dict.items()):
                retain_graph = False if idx == len(layer_output_dict.keys()) - 1 else True
                states = self.compute_states(output[0], output[1], self.params[layer_name]['sig_alpha'], retain_graph=retain_graph)
                scores += self.estimator_dict[layer_name].ood_test(states) / self.layer_num
        scores /= test_size
        return scores

class InDCBFTrainer(pl.LightningModule):
    def __init__(self,model,learning_rate=0.001,weight_decay=0,w_latent=5,w_dyn=5,w_recon=0.5,window_size=5,rtol=5e-6,dt=0.05,**kwargs):
        super(InDCBFTrainer,self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.window_size = window_size
        self.rtol = rtol
        self.dt = dt
        self.w_latent = w_latent
        self.w_dyn = w_dyn
        self.w_recon = w_recon
        self.curr_device = None
        self.save_hyperparameters(ignore=['model'])
        print('----hyper parameters----')
        print(self.hparams)
    
    def forward(self,i,u,x=None):
        return self.model(i,u,x)
    
    def training_step(self, batch, batch_idx):
        i, u = batch
        self.curr_device = i.device

        x,x_tide,i_hat,i_tide = self.model.simulate(i,u,dt=self.dt,window_size=self.window_size,rtol=self.rtol)
        train_loss = self.model.loss_function(i,i_hat,i_tide,x,x_tide)
        print(train_loss)
        train_loss['loss'] = train_loss['loss_latent']*self.w_latent \
               + train_loss['loss_dyn']*self.w_dyn \
               + train_loss['loss_recon']*self.w_recon
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        i, u = batch
        self.curr_device = i.device

        x,x_tide,i_hat,i_tide = self.model.simulate(i,u,dt=self.dt,window_size=self.window_size,rtol=self.rtol)
        val_loss = self.model.loss_function(i,i_hat,i_tide,x,x_tide)
        val_loss['loss'] = val_loss['loss_latent']*self.w_latent \
               + val_loss['loss_dyn']*self.w_dyn \
               + val_loss['loss_recon']*self.w_recon
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        self.sample_images()

    def sample_images(self):          
        i, u = next(iter(self.trainer.datamodule.test_dataloader()))
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
        optimizer = torch.optim.Adam(self.model.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.learning_rate)
        return optimizer