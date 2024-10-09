import torch
import torch.nn.functional
from method.utils import build_mlp
    
class VAE(torch.nn.Module):
    def __init__(self,latent_dim,n_control=2,model="stabilityai/sd-vae-ft-mse",hidden_dim=4*28*28,num_cam=6,freeze_ViT=True):
        super(VAE, self).__init__()
        from diffusers.models import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(model)
        if freeze_ViT:
            for n,p in self.vae.named_parameters():
                p.requires_grad = False
        self.proj = build_mlp([hidden_dim,latent_dim,latent_dim])
        self.rec = torch.nn.Linear(latent_dim,hidden_dim)
        self.fusion = torch.nn.Linear(2*latent_dim+n_control,latent_dim)
        self.num_cam = num_cam
    
    def forward(self,imgs,x_p,u_p):
        B,N,C,H,W = imgs.shape
        rep = self.vae.encode(imgs.reshape(-1,C,H,W))['latent_dist'].mode().reshape(B,N,-1)
        rep = self.proj(rep)
        rep = torch.cat([rep,x_p,u_p.unsqueeze(1).expand(-1,N,-1)],dim=-1)
        return self.fusion(rep)

    def encode(self,imgs,x_p,u_p):
        return self.forward(imgs,x_p,u_p)

    def decode(self,x,trajectory=True):
        if trajectory:
            B,T,N,H = x.shape
            rep = self.rec(x)
            rep = rep.reshape(B*T*N,4,28,28)
            return self.vae.decoder(rep).reshape(B,T,N,3,224,224)
        else:
            B,N,H = x.shape
            rep = self.rec(x)
            rep = rep.reshape(B*N,4,28,28)
            return self.vae.decoder(rep).reshape(B,N,3,224,224)

class ViTEncoder(torch.nn.Module):
    def __init__(self,latent_dim,n_control=2,model="google/vit-base-patch16-224",vit_dim=768,num_cam=6,freeze_ViT=True):
        super(ViTEncoder, self).__init__()
        self.model = model
        if "clip" in model:
            from transformers import CLIPVisionModel
            self.ViT = CLIPVisionModel.from_pretrained(model)
        elif "vc1" in model:
            import vc_models
            from vc_models.models.vit import model_utils
            model,embd_size,model_transforms,model_info = model_utils.load_model(model_utils.VC1_BASE_NAME)
            vit_dim = embd_size
            self.ViT = model
        elif "mvp" in model:
            import mvp
            model = mvp.load("vitb-mae-egosoup")
            self.ViT = model
        else:
            from transformers import ViTModel
            self.ViT = ViTModel.from_pretrained(model)
        if freeze_ViT:
            for n,p in self.ViT.named_parameters():
                if "pooler" not in n:
                    p.requires_grad = False
        self.mlp = build_mlp([vit_dim,latent_dim,latent_dim])
        self.attention = torch.nn.parameter.Parameter(torch.rand((num_cam,latent_dim)))
        self.linear = torch.nn.Linear(2*latent_dim+n_control,latent_dim)
        self.num_cam = num_cam
    
    def forward(self,imgs,x_p,u_p):
        B,N,C,H,W = imgs.shape
        with torch.no_grad():
            if "vc1" in self.model:
                outputs = self.ViT(imgs.reshape(-1,C,H,W))
            else:
                outputs = self.ViT(pixel_values=imgs.reshape(-1,C,H,W))
                outputs = outputs.last_hidden_state.mean(1)
        rep = self.mlp(outputs.reshape(B,N,-1))
        final_rep = torch.cat([rep,x_p,u_p.unsqueeze(1).expand(-1,N,-1)],dim=-1)
        return self.linear(final_rep)

class ResNetEncoderUnfused(torch.nn.Module):
    def __init__(self,latent_dim,n_control=2,model="resnet34",res_dim=1000,num_cam=6):
        super(ResNetEncoderUnfused, self).__init__()
        if "r3m" not in model:
            if "34" in model:
                from torchvision.models import resnet34
                self.ResNet = resnet34(num_classes=res_dim)
            else:
                from torchvision.models import resnet50
                self.ResNet = resnet50(num_classes=res_dim,pretrained=True)
        else:
            from r3m import load_r3m
            if "34" in model:
                self.ResNet = load_r3m("resnet34")
            else:
                self.ResNet = load_r3m("resnet50")
        self.mlp = build_mlp([res_dim+num_cam,latent_dim,latent_dim])
        self.attention = torch.nn.parameter.Parameter(torch.rand((num_cam,latent_dim)))
        self.linear = torch.nn.Linear(2*latent_dim+n_control,latent_dim)
        self.num_cam = num_cam
    
    def forward(self,imgs,x_p,u_p):
        B,N,C,H,W = imgs.shape
        with torch.no_grad():
            outputs = self.ResNet(imgs.reshape(-1,C,H,W))
        pos_encoding = torch.eye(self.num_cam).expand(imgs.shape[0],-1,-1).to(imgs.device)
        rep = torch.cat([outputs.reshape(B,N,-1),pos_encoding],dim=-1)
        rep = self.mlp(rep)
        final_rep = torch.cat([rep,x_p,u_p.unsqueeze(1).expand(-1,N,-1)],dim=-1)
        return self.linear(final_rep)

class SABLASBarrierUnfused(torch.nn.Module):
    def __init__(self,
                 n_control,
                 latent_dim,
                 h_dim = 64,
                #  h_dim = 1024,
                 eps_safe = 1,
                 eps_unsafe = 1,
                 eps_ascent = 1,
                 eps_descent = 1,
                 num_cam = 6,
                 w_safe=1,
                 w_unsafe=1,
                 w_grad=1,
                 w_non_zero=1,
                 w_lambda=1,
                 with_gradient=False,
                 with_nonzero=False,
                 **kwargs
                 ):
        super(SABLASBarrierUnfused, self).__init__()
        modules = []
        # hidden_dims = [latent_dim,h_dim,h_dim,h_dim,1]
        hidden_dims = [latent_dim,h_dim,h_dim,1]
        for i in range(len(hidden_dims)-1):
            modules.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            if not i == len(hidden_dims)-2:
                modules.append(torch.nn.ReLU())
        modules.append(torch.nn.Tanh())
        self.attention = torch.nn.parameter.Parameter(torch.rand((num_cam,latent_dim)))
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

    def forward(self,x,trajectory=True):
        if trajectory:
            B,T,N,H = x.shape
            weight = torch.einsum("btnh,btnh->btn",self.attention.expand(B,T,-1,-1),x)
            x = torch.einsum("btn,btnh->btnh",weight,x).sum(2)
        else:
            B,N,H = x.shape
            weight = torch.einsum("bnh,bnh->bn",self.attention.expand(B,-1,-1),x)
            x = torch.einsum("bn,bnh->bnh",weight,x).sum(2)
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
        gu = torch.einsum('btnha,btna->btnh',g.view(g.shape[0],g.shape[1],g.shape[2],f.shape[-1],self.n_control),u[label == 0,:-1].unsqueeze(2).expand(-1,-1,g.shape[2],-1))
        derive = f + gu
        x_nom = x_safe + derive*0.1
        x_nom = x_nom + (x_tide-x_nom).detach()
        ascent_value = b_safe + (self.forward(x_nom)-b_safe)/0.1
        loss_3 = F.relu(self.eps_ascent-ascent_value).sum(dim=-1).mean()
        output['loss_grad_ascent'] = self.w_grad*loss_3
        output['b_grad_ascent'] = ascent_value.mean()
        return output

class DiscriminatingHyperplaneUnfused(torch.nn.Module):
    def __init__(self,
                 n_control,
                 latent_dim,
                 h_dim = 1024,
                 num_cam = 6,
                 gamma_neg = 5,
                 gamma_pos = 1,
                 **kwargs
                 ):
        super(DiscriminatingHyperplaneUnfused, self).__init__()
        modules = []
        # hidden_dims = [latent_dim,h_dim,h_dim,h_dim,1]
        hidden_dims = [latent_dim,h_dim,h_dim,n_control+1]
        for i in range(len(hidden_dims)-1):
            modules.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            if not i == len(hidden_dims)-2:
                modules.append(torch.nn.ReLU())
        modules.append(torch.nn.Tanh())
        self.attention = torch.nn.parameter.Parameter(torch.rand((num_cam,latent_dim)))
        self.cbf = torch.nn.Sequential(*modules)
        self.n_control = n_control
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self,x, trajectory=True):
        if trajectory:
            B,T,N,H = x.shape
            weight = torch.einsum("btnh,btnh->btn",self.attention.expand(B,T,-1,-1),x)
            x = torch.einsum("btn,btnh->btnh",weight,x).sum(2)
        else:
            B,N,H = x.shape
            weight = torch.einsum("bnh,bnh->bn",self.attention.expand(B,-1,-1),x)
            x = torch.einsum("bn,bnh->bnh",weight,x).sum(2)
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
        value_unsafe = torch.einsum("btc,btc->bt",a_unsafe,u[label == 1]) + b_unsafe
        loss_pos = self.gamma_pos*F.relu(value_safe).mean()
        loss_neg = self.gamma_neg*F.relu(-value_unsafe).mean()
        output = {'loss_pos':loss_pos,'loss_neg':loss_neg}
        return output

class InDCBFBarrier(torch.nn.Module):
    def __init__(self,
                 n_control,
                 latent_dim,
                 num_cam = 6,
                 h_dim = 64,
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
        super(InDCBFBarrier, self).__init__()
        modules = []
        hidden_dims = [latent_dim,h_dim,h_dim,h_dim,1]
        for i in range(len(hidden_dims)-1):
            modules.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            if not i == len(hidden_dims)-2:
                modules.append(torch.nn.ReLU())
        modules.append(torch.nn.Tanh())
        self.attention = torch.nn.parameter.Parameter(torch.rand((num_cam,latent_dim)))
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

    def forward(self,x,trajectory=True):
        if trajectory:
            B,T,N,H = x.shape
            weight = torch.einsum("btnh,btnh->btn",self.attention.expand(B,T,-1,-1),x)
            x = torch.einsum("btn,btnh->btnh",weight,x).sum(2)
        else:
            B,N,H = x.shape
            weight = torch.einsum("bnh,bnh->bn",self.attention.expand(B,-1,-1),x)
            x = torch.einsum("bn,bnh->bnh",weight,x).sum(2)
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
        B,T,N,H = x_g.shape
        weight = torch.einsum("btnh,btnh->btn",self.attention.expand(B,T,-1,-1),x_g)
        x_g_fused = torch.einsum("btn,btnh->btnh",weight,x_g).sum(2)
        b = self.cbf(x_g_fused)
        d_b_safe = torch.autograd.grad(b.mean(),x_g_fused,retain_graph=True)[0]
        with torch.no_grad():
            f, g = ode(x_g)
        gu = torch.einsum('btnha,btna->btnh',g.view(g.shape[0],g.shape[1],g.shape[2],f.shape[-1],self.n_control),u[label == 0].unsqueeze(2).expand(-1,-1,g.shape[2],-1))
        dx = torch.einsum("btn,btnh->btnh",weight,(f + gu)).sum(2)
        ascent_value = torch.einsum('bth,bth->bt', d_b_safe, dx)
        loss_3 = F.relu(self.eps_ascent - ascent_value.unsqueeze(-1) - b).sum(dim=-1).mean()
        output['loss_grad_ascent'] = self.w_grad*loss_3
        output['b_grad_ascent'] = (ascent_value.unsqueeze(-1) + b).mean()
        return output

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

class iDBFDynamics(torch.nn.Module):
    def __init__(self,n_control,device,model,num_cam=6,latent_dim=256,h_dim=1024):
        super(iDBFDynamics, self).__init__()
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