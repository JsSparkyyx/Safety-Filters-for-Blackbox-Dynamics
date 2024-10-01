import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import torch
from torchdiffeq import odeint
from tqdm import trange
from torchvision.utils import save_image
from method.utils import build_mlp, NeuralODE, BlackBoxNODE
from method.backbones import VAE, ViTEncoder, ViTAttentionEncoder, ResNetEncoder, ResNetEncoderUnfused, QuantityPermutationInvariantEncoder
import cvxpy as cp

class InDCBFDynamicsWithRec(torch.nn.Module):
    def __init__(self,n_control,device,model,num_cam=6,latent_dim=256,h_dim=1024):
        super(InDCBFDynamicsWithRec, self).__init__()
        self.latent_dim = latent_dim
        self.device = device
        self.vae = VAE(latent_dim,model=model)
        self.ode = NeuralODE([latent_dim,h_dim,h_dim,latent_dim],
                             [latent_dim,h_dim,h_dim,latent_dim*n_control])
        self.n_control = n_control
        self.num_cam = num_cam

    def forward(self,i,u_p,x_p,u_ref,barrier):
        u_ref = u_ref.view(-1).cpu().numpy()
        x = self.vae(i,x_p,u_p)
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
        x_init = torch.zeros(i.shape[0],self.num_cam,self.latent_dim).to(self.device)
        u = torch.cat([u[:,0,:].unsqueeze(1),u],dim=1)
        x = self.vae(i[:,0,:],x_init,u[:,0])
        x_tide = x
        xs = [x]
        x_tides = [x_tide]
        for k in trange(1,i.shape[1]):
            if k % window_size == 0:
                x_tide = x
            def odefunc(t,state):
                f, g = self.ode(state)
                gu = torch.einsum("bnha,bna->bnh",g.view(g.shape[0],g.shape[1],-1,self.n_control),u[:,k+1].unsqueeze(1).expand(-1,g.shape[1],-1))
                return f + gu
            timesteps = torch.Tensor([0,dt]).to(self.device)
            x_tide = odeint(odefunc,x_tide,timesteps,rtol=rtol)[1,:,:]
            x = self.vae(i[:,k,:],x,u[:,k])
            xs.append(x)
            x_tides.append(x_tide)
        xs = torch.stack(xs,dim=1)
        x_tides = torch.stack(x_tides,dim=1)
        i_hat = self.vae.decode(xs)
        i_tide = self.vae.decode(x_tides)
        return  (xs,x_tides,i_hat,i_tide)
    
    def loss_function(self,i,i_hat,i_tide,x,x_tide):
        loss_latent = F.mse_loss(x,x_tide)
        loss_dyn = F.mse_loss(i,i_tide)
        loss_recon = F.mse_loss(i,i_hat)
        return {'loss_latent': loss_latent,'loss_dyn': loss_dyn,'loss_recon': loss_recon}
    
class InDCBFDynamics(torch.nn.Module):
    def __init__(self,n_control,device,model,num_cam=6,latent_dim=256,h_dim=1024):
        super(InDCBFDynamics, self).__init__()
        self.latent_dim = latent_dim
        self.device = device
        if "resnet" in model:
            self.encoder = ResNetEncoderUnfused(latent_dim,model=model)
        else:
            self.vae = ViTEncoder(latent_dim,model=model)
        self.ode = NeuralODE([latent_dim,h_dim,h_dim,latent_dim],
                             [latent_dim,h_dim,h_dim,latent_dim*n_control])
        self.n_control = n_control
        self.num_cam = num_cam

    def forward(self,i,u_p,x_p,u_ref,barrier):
        u_ref = u_ref.view(-1).cpu().numpy()
        x = self.vae(i,x_p,u_p)
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
        x_init = torch.zeros(i.shape[0],self.num_cam,self.latent_dim).to(self.device)
        u = torch.cat([u[:,0,:].unsqueeze(1),u],dim=1)
        x = self.vae(i[:,0,:],x_init,u[:,0])
        x_tide = x
        xs = [x]
        x_tides = [x_tide]
        for k in trange(1,i.shape[1]):
            if k % window_size == 0:
                x_tide = x
            def odefunc(t,state):
                f, g = self.ode(state)
                gu = torch.einsum("bnha,bna->bnh",g.view(g.shape[0],g.shape[1],-1,self.n_control),u[:,k+1].unsqueeze(1).expand(-1,g.shape[1],-1))
                return f + gu
            timesteps = torch.Tensor([0,dt]).to(self.device)
            x_tide = odeint(odefunc,x_tide,timesteps,rtol=rtol)[1,:,:]
            x = self.vae(i[:,k,:],x,u[:,k])
            xs.append(x)
            x_tides.append(x_tide)
        xs = torch.stack(xs,dim=1)
        x_tides = torch.stack(x_tides,dim=1)
        return  (xs,x_tides)
    
    def loss_function(self,x,x_tide):
        loss_latent = F.mse_loss(x,x_tide)
        return {'loss_latent': loss_latent}
    
class SABLASUnFusedDynamics(torch.nn.Module):
    def __init__(self,n_control,device,model,dynamics=None,num_cam=6,latent_dim=256,h_dim=1024):
        super(SABLASUnFusedDynamics, self).__init__()
        self.latent_dim = latent_dim
        self.device = device
        self.dynamics = dynamics
        self.encoder = ViTEncoder(latent_dim,model=model)
        self.ode = NeuralODE([latent_dim,h_dim,h_dim,latent_dim],
                             [latent_dim,h_dim,h_dim,latent_dim*n_control])
        self.n_control = n_control
        self.num_cam = num_cam

    def simulate(self,i,u,dt=0.1,window_size=5,rtol=5e-6):
        x_init = torch.zeros(i.shape[0],self.num_cam,self.latent_dim).to(self.device)
        u = torch.cat([u[:,0,:].unsqueeze(1),u],dim=1)
        x = self.encoder(i[:,0,:],x_init,u[:,0])
        xs = [x]
        x_tides = None
        for k in trange(1,i.shape[1]):
            x = self.encoder(i[:,k,:],x,u[:,k])
            xs.append(x)
        xs = torch.stack(xs,dim=1)
        return (xs,x_tides)
    
    def loss_function(self,x,x_tide):
        loss_latent = F.mse_loss(x,x_tide)
        return {'loss_latent': loss_latent}

class InDCBFAttentionDynamics(torch.nn.Module):
    def __init__(self,n_control,device,model,latent_dim=256,h_dim=1024):
        super(InDCBFAttentionDynamics, self).__init__()
        self.latent_dim = latent_dim
        self.device = device
        if "resnet" in model:
            self.encoder = ResNetEncoder(latent_dim,model=model)
        else:
            self.encoder = ViTAttentionEncoder(latent_dim,model=model)
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
        return (xs,x_tides)
    
    def loss_function(self,x,x_tide):
        loss_latent = F.mse_loss(x,x_tide)
        return {'loss_latent': loss_latent}

class SABLASDynamics(torch.nn.Module):
    def __init__(self,n_control,device,model,dynamics=None,num_cam=6,latent_dim=256,h_dim=1024):
        super(SABLASDynamics, self).__init__()
        self.latent_dim = latent_dim
        self.device = device
        self.dynamics = dynamics
        if "resnet" in model:
            self.encoder = ResNetEncoder(latent_dim,model=model)
        else:
            self.encoder = ViTAttentionEncoder(latent_dim,model=model)
        # self.ode = BlackBoxNODE([latent_dim+n_control,h_dim,h_dim,h_dim,latent_dim])
        self.ode = NeuralODE([latent_dim,h_dim,h_dim,latent_dim],
                             [latent_dim,h_dim,h_dim,latent_dim*n_control])
        self.n_control = n_control
        self.num_cam = num_cam

    def simulate(self,i,u,dt=0.1,window_size=5,rtol=5e-6):
        x_init = torch.zeros(i.shape[0],self.latent_dim).to(self.device)
        u = torch.cat([u[:,0,:].unsqueeze(1),u],dim=1)
        x = self.encoder(i[:,0,:],x_init,u[:,0])
        xs = [x]
        x_tides = None
        for k in trange(1,i.shape[1]):
            x = self.encoder(i[:,k,:],x,u[:,k])
            xs.append(x)
        xs = torch.stack(xs,dim=1)
        return (xs,x_tides)
    
    def loss_function(self,x,x_tide):
        loss_latent = F.mse_loss(x,x_tide)
        return {'loss_latent': loss_latent}
    
class HyperplaneEncoder(torch.nn.Module):
    def __init__(self,n_control,device,model,dynamics=None,num_cam=6,latent_dim=256,h_dim=1024):
        super(HyperplaneEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.device = device
        self.dynamics = dynamics
        if "resnet" in model:
            self.encoder = ResNetEncoder(latent_dim,model=model)
        else:
            self.encoder = ViTAttentionEncoder(latent_dim,model=model)
        self.n_control = n_control
        self.num_cam = num_cam

    def simulate(self,i,u):
        x_init = torch.zeros(i.shape[0],self.latent_dim).to(self.device)
        u = torch.cat([u[:,0,:].unsqueeze(1),u],dim=1)
        x = self.encoder(i[:,0,:],x_init,u[:,0])
        xs = [x]
        for k in trange(1,i.shape[1]):
            x = self.encoder(i[:,k,:],x,u[:,k])
            xs.append(x)
        xs = torch.stack(xs,dim=1)
        return xs
    
class HyperplaneEncoderUnfused(torch.nn.Module):
    def __init__(self,n_control,device,model,dynamics=None,num_cam=6,latent_dim=256,h_dim=1024):
        super(HyperplaneEncoderUnfused, self).__init__()
        self.latent_dim = latent_dim
        self.device = device
        self.dynamics = dynamics
        self.encoder = ViTEncoder(latent_dim,model=model)
        self.n_control = n_control
        self.num_cam = num_cam

    def simulate(self,i,u):
        x_init = torch.zeros(i.shape[0],self.num_cam,self.latent_dim).to(self.device)
        u = torch.cat([u[:,0,:].unsqueeze(1),u],dim=1)
        x = self.encoder(i[:,0,:],x_init,u[:,0])
        xs = [x]
        for k in trange(1,i.shape[1]):
            x = self.encoder(i[:,k,:],x,u[:,k])
            xs.append(x)
        xs = torch.stack(xs,dim=1)
        return xs
    
class GroundTruthiDBFDynamics(torch.nn.Module):
    def __init__(self,n_control,device,latent_dim=256,h_dim=64):
        super(GroundTruthiDBFDynamics, self).__init__()
        self.latent_dim = latent_dim
        self.device = device
        self.encoder = QuantityPermutationInvariantEncoder(latent_dim=latent_dim)
        self.ode = NeuralODE([latent_dim,h_dim,h_dim,latent_dim],
                             [latent_dim,h_dim,h_dim,latent_dim*n_control])
        self.n_control = n_control

    def simulate(self,i,u,dt=0.1,window_size=5,rtol=5e-6):
        x = self.encoder(i[:,0,:])
        x_tide = x
        xs = [x]
        x_tides = [x_tide]
        for k in trange(1,i.shape[1]):
            if k % window_size == 1:
                x_tide = x
            def odefunc(t,state):
                f, g = self.ode(state)
                gu = torch.bmm(g.view(g.shape[0],-1,self.n_control),u[:,k].unsqueeze(-1))
                return f + gu.squeeze(-1)
            timesteps = torch.Tensor([0,dt]).to(self.device)
            x_tide = odeint(odefunc,x_tide,timesteps,rtol=rtol)[1,:,:]
            x = self.encoder(i[:,k,:])
            xs.append(x)
            x_tides.append(x_tide)
        xs = torch.stack(xs,dim=1)
        x_tides = torch.stack(x_tides,dim=1)
        return (xs,x_tides)
    
    def loss_function(self,x,x_tide):
        loss_latent = F.mse_loss(x,x_tide)
        return {'loss_latent': loss_latent}

class GroundTruthSABLASDynamics(torch.nn.Module):
    def __init__(self,n_control,device,dynamics=None,num_cam=6,latent_dim=256,h_dim=1024):
        super(GroundTruthSABLASDynamics, self).__init__()
        self.latent_dim = latent_dim
        self.device = device
        self.dynamics = dynamics
        self.encoder = QuantityPermutationInvariantEncoder(latent_dim=latent_dim)
        # self.ode = BlackBoxNODE([latent_dim+n_control,h_dim,h_dim,h_dim,latent_dim])
        self.ode = NeuralODE([latent_dim,h_dim,h_dim,latent_dim],
                             [latent_dim,h_dim,h_dim,latent_dim*n_control])
        self.n_control = n_control
        self.num_cam = num_cam

    def simulate(self,i,u,dt=0.1,window_size=5,rtol=5e-6):
        x = self.encoder(i[:,0,:])
        u = torch.cat([u[:,0,:].unsqueeze(1),u],dim=1)
        xs = [x]
        x_tides = None
        for k in trange(1,i.shape[1]):
            x = self.encoder(i[:,k,:])
            xs.append(x)
        xs = torch.stack(xs,dim=1)
        return (xs,x_tides)
    
    def loss_function(self,x,x_tide):
        loss_latent = F.mse_loss(x,x_tide)
        return {'loss_latent': loss_latent}

class GroundTruthDHSDynamics(torch.nn.Module):
    def __init__(self,n_control,device,dynamics=None,num_cam=6,latent_dim=256,h_dim=1024):
        super(GroundTruthDHSDynamics, self).__init__()
        self.latent_dim = latent_dim
        self.device = device
        self.dynamics = dynamics
        self.encoder = QuantityPermutationInvariantEncoder(latent_dim=latent_dim)
        self.n_control = n_control
        self.num_cam = num_cam

    def simulate(self,i,u):
        u = torch.cat([u[:,0,:].unsqueeze(1),u],dim=1)
        x = self.encoder(i[:,0,:])
        xs = [x]
        for k in trange(1,i.shape[1]):
            x = self.encoder(i[:,k,:])
            xs.append(x)
        xs = torch.stack(xs,dim=1)
        return xs