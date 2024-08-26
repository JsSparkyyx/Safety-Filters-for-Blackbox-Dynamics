import torch
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
        if "clip" in model:
            from transformers import CLIPVisionModel
            self.ViT = CLIPVisionModel.from_pretrained(model)
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
            outputs = self.ViT(pixel_values=imgs.reshape(-1,C,H,W))
        rep = self.mlp(outputs.pooler_output.reshape(B,N,-1))
        final_rep = torch.cat([rep,x_p,u_p.unsqueeze(1).expand(-1,N,-1)],dim=-1)
        return self.linear(final_rep)
    
class ViTAttentionEncoder(torch.nn.Module):
    def __init__(self,latent_dim,n_control=2,model="google/vit-base-patch16-224",vit_dim=768,num_cam=6,freeze_ViT=True):
        super(ViTAttentionEncoder, self).__init__()
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
        self.mlp = build_mlp([vit_dim+num_cam,latent_dim,latent_dim])
        self.attention = torch.nn.parameter.Parameter(torch.rand((num_cam,latent_dim)))
        self.linear = torch.nn.Linear(2*latent_dim+n_control,latent_dim)
        self.num_cam = num_cam
    
    def forward(self,imgs,x_p,u_p):
        B,N,C,H,W = imgs.shape
        pos_encoding = torch.eye(self.num_cam).expand(imgs.shape[0],-1,-1).to(imgs.device)
        with torch.no_grad():
            if "vc1" in self.model:
                outputs = self.ViT(imgs.reshape(-1,C,H,W))
            else:
                outputs = self.ViT(pixel_values=imgs.reshape(-1,C,H,W))
                outputs = outputs.pooler_output
        rep = torch.cat([outputs.reshape(B,N,-1),pos_encoding],dim=-1)
        rep = self.mlp(rep)
        weight = torch.einsum("bch,bch->bc",self.attention.expand(B,-1,-1),rep)
        final_rep = torch.einsum("bn,bnh->bnh",weight,rep).sum(1)
        final_rep = torch.cat([final_rep,x_p,u_p],dim=-1)
        return self.linear(final_rep)
    
class ResNetEncoder(torch.nn.Module):
    def __init__(self,latent_dim,n_control=2,model="resnet34",res_dim=1000,num_cam=6):
        super(ResNetEncoder, self).__init__()
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
        weight = torch.einsum("bch,bch->bc",self.attention.expand(B,-1,-1),rep)
        final_rep = torch.einsum("bn,bnh->bnh",weight,rep).sum(1)
        final_rep = torch.cat([final_rep,x_p,u_p],dim=-1)
        return self.linear(final_rep)