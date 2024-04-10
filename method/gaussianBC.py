import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

class GaussianBC(torch.nn.Module):
    def __init__(self,H,W,C,latent_dim,action_dim):
        super(GaussianBC, self).__init__()
        self.fc1 = torch.nn.Linear(H*W*C, latent_dim)
        self.fc2 = torch.nn.Linear(latent_dim, latent_dim)
        self.mean_layer = torch.nn.Linear(latent_dim, action_dim)
        self.variance_layer = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, action_dim),
            torch.nn.Softplus()  # Softplus activation ensures the variance is always positive
        )

    def forward(self, x):
        x = x.view(x.shape[0],-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.mean_layer(x)
        var = self.variance_layer(x)
        return mu, var
    
    def loss_function(self,u,mu,var):
        loss = F.gaussian_nll_loss(u,mu,var)
        # loss = torch.mean(0.5 * torch.log(var) + 0.5 * torch.square((u - mu) / var) + 0.5 * torch.log(torch.ones_like(var)*2*np.pi))
        return {'loss': loss}
    
    def generate(self, state):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, var = self.forward(state)
        std = torch.exp(0.5 * torch.log(var))
        eps = torch.randn_like(std)
        return eps * std + mu
    

class GaussianTrainer(pl.LightningModule):
    def __init__(self,model,args):
        super(GaussianTrainer, self).__init__()
        self.model = model
        self.args = args
        self.curr_device = None
    
    def forward(self, state):
        return self.model(state)
    
    def training_step(self, batch, batch_idx):
        state, u = batch
        self.curr_device = state.device

        mu, var = self.forward(state)
        train_loss = self.model.loss_function(u,mu,var)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        state, u = batch
        self.curr_device = state.device

        mu, var = self.forward(state)
        val_loss = self.model.loss_function(u,mu,var)
        prediction = self.model.generate(state)
        mse_loss = F.mse_loss(prediction,u)
        du_loss = torch.mean(mu-u)
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        self.log_dict({f"mse_loss": mse_loss.item()}, sync_dist=True)
        self.log_dict({f"du_loss": du_loss.item()}, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                               lr=self.args['learning_rate'],
                               weight_decay=self.args['weight_decay'])
        return optimizer