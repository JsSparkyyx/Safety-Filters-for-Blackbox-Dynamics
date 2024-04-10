import torch
import torch.nn.functional as F
import pytorch_lightning as pl

class GaussianBC(torch.nn.Module):
    def __init__(self,H,W,C,latent_dim,action_dim):
        super(GaussianBC, self).__init__()
        modules = torch.nn.ModuleList()
        hidden_dims = None
        if hidden_dims is None:
            hidden_dims = [256,512]
        
        in_dim = H*W*C
        for h_dim in hidden_dims:
            modules.append(
                torch.nn.Sequential(
                    torch.nn.Linear(in_dim,h_dim),
                    torch.nn.ReLU())
            )
            in_dim = h_dim

        self.encoder = torch.nn.Sequential(*modules)
        self.fc_mu = torch.nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = torch.nn.Linear(hidden_dims[-1], latent_dim)
        self.decoder = torch.nn.Linear(latent_dim,action_dim)

    def encode(self,state):
        state = state.view(state.shape[0],-1)
        result = self.encoder(state)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self,state):
        mu, log_var = self.encode(state)
        z = self.reparameterize(mu, log_var)
        return [self.decoder(z), mu, log_var]
    
    def loss_function(self,u_recons,u,log_var,mu):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons_loss =F.mse_loss(u_recons, u)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
    
    def generate(self, state):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(state)[0]
    

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

        u_recons,log_var,mu = self.forward(state)
        train_loss = self.model.loss_function(u_recons,u,log_var,mu)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']
    
    def validation_step(self, batch, batch_idx):
        state, u = batch
        self.curr_device = state.device

        u_recons,log_var,mu = self.forward(state)
        val_loss = self.model.loss_function(u_recons,u,log_var,mu)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                               lr=self.args['learning_rate'],
                               weight_decay=self.args['weight_decay'])
        return optimizer