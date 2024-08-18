import pytorch_lightning as pl
import torch
import os
import numpy as np
from torchvision.utils import save_image

class InDCBFTrainerWithRec(pl.LightningModule):
    def __init__(self,
                 model,
                 barrier = None,
                 learning_rate=0.001,
                 weight_decay=0,
                 w_barrier=2,
                 w_latent=1,
                 window_size=5,
                 rtol=5e-6,
                 dt=0.05,
                 with_dynamic=True,
                 train_barrier=False,
                 **kwargs):
        super(InDCBFTrainerWithRec,self).__init__()
        self.model = model
        self.barrier = barrier
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.window_size = window_size
        self.rtol = rtol
        self.dt = dt
        self.w_latent = w_latent
        self.w_barrier = w_barrier
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

        xs,x_tides,i_hat,i_tide = self.model.simulate(i,u,dt=self.dt,window_size=self.window_size,rtol=self.rtol)
        train_loss = self.model.loss_function(i,i_hat,i_tide,xs,x_tides)
        train_loss['loss'] = 0
        if self.with_dynamic:
            train_loss['loss'] += train_loss['loss_latent']*self.w_latent
            train_loss['loss'] += train_loss['loss_dyn']*self.w_latent
            train_loss['loss'] += train_loss['loss_recon']*self.w_latent
        if self.train_barrier:
            output = self.barrier.loss_function(xs,label,u,self.model.ode)
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

        xs,x_tides,i_hat,i_tide = self.model.simulate(i,u,dt=self.dt,window_size=self.window_size,rtol=self.rtol)
        val_loss = self.model.loss_function(i,i_hat,i_tide,xs,x_tides)
        val_loss['loss'] = 0
        if self.with_dynamic:
            val_loss['loss'] += val_loss['loss_latent']*self.w_latent
            val_loss['loss'] += val_loss['loss_dyn']*self.w_latent
            val_loss['loss'] += val_loss['loss_recon']*self.w_latent
        if self.train_barrier:
            output = self.barrier.loss_function(xs,label,u,self.model.ode)
            val_loss['loss_safe'] = output['loss_safe']
            val_loss['loss_unsafe'] = output['loss_unsafe']
            val_loss['loss_grad_ascent'] = output['loss_grad_ascent']
            val_loss['loss'] += output['loss_safe']*self.w_barrier+output['loss_unsafe']*self.w_barrier
            val_loss['loss'] += output['loss_grad_ascent']*self.w_barrier
            self.log_dict({'val_b_safe':output['b_safe'],'val_b_unsafe':output['b_unsafe']},sync_dist=True)
            self.log_dict({'val_b_grad_ascent':output['b_grad_ascent']},sync_dist=True)
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        self.sample_states()
        # pass

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
        params = [{"params":self.model.parameters(),"lr":self.learning_rate,"weight_decay":self.learning_rate}]
        if self.train_barrier:
            params.append({"params":self.barrier.parameters(),"lr":self.learning_rate,"weight_decay":self.learning_rate}
                                    )
        optimizer = torch.optim.Adam(params)
        return optimizer
    
class InDCBFTrainer(pl.LightningModule):
    def __init__(self,
                 model,
                 barrier = None,
                 learning_rate=0.001,
                 weight_decay=0,
                 w_barrier=2,
                 w_latent=1,
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

        xs,x_tides = self.model.simulate(i,u,dt=self.dt,window_size=self.window_size,rtol=self.rtol)
        train_loss = self.model.loss_function(xs,x_tides)
        train_loss['loss'] = 0
        if self.with_dynamic:
            train_loss['loss'] += train_loss['loss_latent']*self.w_latent
        if self.train_barrier:
            output = self.barrier.loss_function(xs,label,u,self.model.ode)
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

        xs,x_tides = self.model.simulate(i,u,dt=self.dt,window_size=self.window_size,rtol=self.rtol)
        val_loss = self.model.loss_function(xs,x_tides)
        val_loss['loss'] = 0
        if self.with_dynamic:
            val_loss['loss'] += val_loss['loss_latent']*self.w_latent
        if self.train_barrier:
            output = self.barrier.loss_function(xs,label,u,self.model.ode)
            val_loss['loss_safe'] = output['loss_safe']
            val_loss['loss_unsafe'] = output['loss_unsafe']
            val_loss['loss_grad_ascent'] = output['loss_grad_ascent']
            val_loss['loss'] += output['loss_safe']*self.w_barrier+output['loss_unsafe']*self.w_barrier
            val_loss['loss'] += output['loss_grad_ascent']*self.w_barrier
            self.log_dict({'val_b_safe':output['b_safe'],'val_b_unsafe':output['b_unsafe']},sync_dist=True)
            self.log_dict({'val_b_grad_ascent':output['b_grad_ascent']},sync_dist=True)
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        # self.sample_states()
        pass

    def sample_states(self):          
        i, u, label = next(iter(self.trainer.datamodule.test_dataloader()))
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
    
class SABLASTrainer(pl.LightningModule):
    def __init__(self,
                 model,
                 barrier = None,
                 learning_rate=0.001,
                 weight_decay=0,
                 w_barrier=2,
                 w_latent=1,
                 window_size=5,
                 rtol=5e-6,
                 dt=0.05,
                 with_dynamic=True,
                 train_barrier=False,
                 **kwargs):
        super(SABLASTrainer,self).__init__()
        self.model = model
        self.barrier = barrier
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.window_size = window_size
        self.rtol = rtol
        self.dt = dt
        self.w_latent = w_latent
        self.w_barrier = w_barrier
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

        x,x_tide = self.model.simulate(i,u,dt=self.dt,window_size=self.window_size,rtol=self.rtol)
        # train_loss = self.model.loss_function(x,x_tide)
        train_loss = {}
        train_loss['loss'] = 0
        # if self.with_dynamic:
        #     train_loss['loss'] += train_loss['loss_latent']*self.w_latent
        if self.train_barrier:
            output = self.barrier.loss_function(x,label,u,self.model.ode,self.model.dynamics)
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
        # val_loss = self.model.loss_function(x,x_tide)
        val_loss = {}
        val_loss['loss'] = 0
        # if self.with_dynamic:
        #     val_loss['loss'] += val_loss['loss_latent']*self.w_latent
        if self.train_barrier:
            output = self.barrier.loss_function(x,label,u,self.model.ode,self.model.dynamics)
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
    
    def on_train_epoch_start(self):
        if self.current_epoch >= 5:
            if self.barrier.w_grad <= 0.01:
                self.barrier.w_grad += 0.001

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
    
class HyperplaneTrainer(pl.LightningModule):
    def __init__(self,
                 model,
                 barrier = None,
                 learning_rate=0.001,
                 weight_decay=0,
                 **kwargs):
        super(HyperplaneTrainer,self).__init__()
        self.model = model
        self.barrier = barrier
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.curr_device = None
        self.save_hyperparameters(ignore=['model','barrier'])
        # print('----hyper parameters----')
        # print(self.hparams)
    
    def forward(self,i,u,x=None):
        return self.model(i,u,x)
    
    def training_step(self, batch, batch_idx):
        i, u, label = batch
        self.curr_device = i.device

        x = self.model.simulate(i,u)
        train_loss = {}
        train_loss['loss'] = 0
        output = self.barrier.loss_function(x,label,u)
        train_loss['loss_pos'] = output['loss_pos']
        train_loss['loss_neg'] = output['loss_neg']
        train_loss['loss'] += output['loss_pos']+output['loss_neg']
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        i, u, label = batch
        self.curr_device = i.device

        x = self.model.simulate(i,u)
        val_loss = {}
        val_loss['loss'] = 0
        output = self.barrier.loss_function(x,label,u)
        val_loss['loss_pos'] = output['loss_pos']
        val_loss['loss_neg'] = output['loss_neg']
        val_loss['loss'] += output['loss_pos']+output['loss_neg']
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        
    def configure_optimizers(self):
        params = [{"params":self.model.parameters(),"lr":self.learning_rate,"weight_decay":self.learning_rate}]
        params.append({"params":self.barrier.parameters(),"lr":self.learning_rate,"weight_decay":self.learning_rate}
                                    )
        optimizer = torch.optim.Adam(params)
        return optimizer