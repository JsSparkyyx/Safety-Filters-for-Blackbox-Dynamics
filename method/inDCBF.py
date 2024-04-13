import torch.nn.functional as F
import torch
import pytorch_lightning as pl
from torch import nn
from torchdiffeq import odeint
from tqdm import trange, tqdm
import os
from torchvision.utils import save_image
import numpy as np

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
        self.num_f = len(params_f-1)
        self.num_g = len(params_g-1)

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
    def __init__(self,C,n_control,model_ref,device,params=None,latent_dim=256,h_dim=256,gamma=0.5,sample_size=200):
        super(InDCBFController, self).__init__()
        self.model_ref = model_ref
        self.latent_dim = latent_dim
        self.device = device
        self.vae = VAE(C,n_control,latent_dim)
        self.ode = NeuralODE([latent_dim,h_dim,h_dim,latent_dim],
                             [latent_dim,h_dim,h_dim,latent_dim*n_control])
        self.n_control = n_control
        self.sample_size = sample_size
        self.params = params
        self.init_nac_estimator()

    def simulate(self,i,u):
        x_init = torch.zeros(i.shape[0],self.latent_dim).to(self.device)
        u = torch.cat([u[:,0,:].unsqueeze(1),u],dim=1)
        x = self.vae(i[:,0,:],x_init,u[:,0])
        x_tide = x
        xs = [x]
        x_tides = [x_tide]
        for k in trange(1,i.shape[1]):
            if k % 5 == 1:
                x_tide = x
            def odefunc(t,state):
                f, g = self.ode(state)
                gu = torch.bmm(g.view(g.shape[0],-1,self.n_control),u[:,k+1].unsqueeze(-1))
                return f + gu.squeeze(-1)
            timesteps = torch.Tensor([k*0.05,(k+1)*0.05]).to(self.device)
            x_tide = odeint(odefunc,x_tide,timesteps,rtol=5e-6)[1,:,:]
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
            x = torch.zeros(i.shape[0],self.latent_dim)
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

    def init_nac_estimator(self,params):
        self.estimator = {}
        x = torch.rand(2,128).to(self.device)
        layer_state_dict = self.ode.get_layer_output(x)
        for n, state in layer_state_dict.items():
            self.estimator[n] = Estimator(state.shape[1], params[n]['M'], params[n]['O'], self.device)

    def save_neural_states(self,data_loader):
        self.eval()
        print('Accessing neural states')
        for i, u in tqdm(data_loader):
            x_init = torch.zeros(i.shape[0],self.latent_dim).to(self.device)
            u = torch.cat([u[:,0,:].unsqueeze(1),u],dim=1)
            x = self.vae(i[:,0,:],x_init,u[:,0])
            for k in trange(1,i.shape[1]):
                x = self.vae(i[:,k,:],x,u[:,k])
                xs.append(x)
            xs = torch.cat(xs,0)
            layer_output_dict = self.ode.get_layer_output(xs)
            layer_state_dict = {}
            for idx, layer_name, output in enumerate(layer_output_dict.items()):
                retain_graph = False if idx == len(layer_output_dict.keys()) - 1 else True
                states = self.compute_states(output[0], output[1], self.params[layer_name]['sig_alpha'], retain_graph=retain_graph)
                if len(states) > 0:
                    self.estimator_dict[layer_name].update(states)

    def compute_states(self, h, outputs,layer_name, sig_alpha, retain_graph=False, **kwargs):
        baseline = torch.zeros_like(outputs)

        loss = F.mse_loss(baseline,outputs)
        layer_grad = torch.autograd.grad(loss.sum(), h, create_graph=False,
                                        retain_graph=retain_graph, **kwargs)[0]
        states = sigmoid(h*layer_grad, sig_alpha=sig_alpha)
        return states

    def nac_filter(self,u):

        return

def sigmoid(x, sig_alpha=1.0):
    """
    sig_alpha is the steepness controller (larger denotes steeper)
    """
    return 1 / (1 + torch.exp(-sig_alpha * x))

def logspace(base=10, num=100):
    num = int(num / 2)
    x = np.linspace(1, np.sqrt(base), num=num)
    x_l = np.emath.logn(base, x)
    x_r = (1 - x_l)[::-1]
    x = np.concatenate([x_l[:-1], x_r])
    x[-1] += 1e-2
    return torch.from_numpy(np.append(x, 1.2))

class Estimator(object):
    def __init__(self, neuron_num, M=1000, O=1, device=None):
        assert O > 0, 'minumum activated number O should > (or =) 1'
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.M, self.O, self.N = M, O, neuron_num
        # self.thresh = torch.linspace(0., 1.01, M).view(M, -1).repeat(1, neuron_num).to(self.device)
        self.thresh = logspace(1e3, M).view(M, -1).repeat(1, neuron_num).to(self.device)
        self.t_act = torch.zeros(M - 1, neuron_num).to(self.device)  # current activations under each thresh
        self.n_coverage = None

    def add(self, other):
        # check if other is an Estimator object
        assert (self.M == other.M) and (self.N == other.N)
        self.t_act += other.t_act

    def update(self, states):
        # thresh -> [num_t, num_n] -> [1, num_t, num_n] ->compare-> [num_data, num_t, num_n]
        # states -> [num_data, num_n] -> [num_data, 1, num_n] ->compare-> ...
        """
        Here is the example to check this code:
            k = 10
            states = torch.rand(2, 8)
            thresh = torch.linspace(0., 1., M).view(M, -1).repeat(1, 8)
            b_act = (states.unsqueeze(1) >= thresh[:M - 1].unsqueeze(0)) & \
                            (states.unsqueeze(1) < thresh[1:M].unsqueeze(0))

            b_act.sum(dim=1)
        """
        with torch.no_grad():
            b_act = (states.unsqueeze(1) >= self.thresh[:self.M - 1].unsqueeze(0)) & \
                    (states.unsqueeze(1) < self.thresh[1:self.M].unsqueeze(0))
            b_act = b_act.sum(dim=0)  # [num_t, num_n]
            # print(states.shape[0], b_act.sum(0)[:3])

            self.t_act += b_act  # current activation times under each interval

    def get_score(self, method="avg"):
        t_score = torch.min(self.t_act / self.O, torch.ones_like(self.t_act))  # [num_t, num_n]
        coverage = (t_score.sum(dim=0)) / self.M  # [num_n]
        if method == "norm2":
            coverage = coverage.norm(p=1).cpu()
        elif method == "avg":
            coverage = coverage.mean().cpu()

        t_cov = t_score.mean(dim=1).cpu().numpy()  # for simplicity
        self.n_coverage = t_score  # [num_t, num_n]
        return np.append(t_cov, 0), coverage

    def ood_test(self, states, method="avg"):
        # thresh -> [num_t, num_n] -> [1, num_t, num_n] ->compare-> [num_data, num_t, num_n]
        # states -> [num_data, num_n] -> [num_data, 1, num_n] ->compare-> ...
        b_act = (states.unsqueeze(1) >= self.thresh[:self.M - 1].unsqueeze(0)) & \
                (states.unsqueeze(1) < self.thresh[1:self.M].unsqueeze(0))
        scores = (b_act * self.n_coverage.unsqueeze(0)).sum(dim=1)  # [num_data, num_n]
        if method == "avg":
            scores = scores.mean(dim=1)
        return scores

    @property
    def states(self):
        return {
            "thresh": self.thresh.cpu(),
            "t_act": self.t_act.cpu()
        }

    def load(self, state_dict, zero_corner=True):
        self.thresh = state_dict["thresh"].to(self.device)
        self.t_act = state_dict["t_act"].to(self.device)

    def clear(self):
        self.t_act = torch.zeros(self.M - 1, self.N).to(self.device)  # current activations under each thresh

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
                                       f"recon_decode_Epoch_{self.current_epoch}.png"))
        save_image(i_tide.data[0],
                          os.path.join(self.logger.log_dir , 
                                       "ReconDynamic", 
                                       f"recon_dynamic_Epoch_{self.current_epoch}.png"))
        save_image(i.data[0],
                          os.path.join(self.logger.log_dir , 
                                       "Samples", 
                                       f"sample_Epoch_{self.current_epoch}.png"))
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
                               lr=self.args['learning_rate'],
                               weight_decay=self.args['weight_decay'])
        return optimizer