import argparse
import yaml
import sys
sys.path.append('D:/Code/WashU/gcg')
sys.path.append('./system')
from system.square_simu import SquareEnv
import numpy as np
from method.inDCBF import InDCBFTrainer, InDCBFController
from method.gaussianBC import GaussianTrainer, GaussianBC
from method.baselineController import BCDensityFilter, ReferenceController
from dataset import RACCARDataset
import torch
from tqdm import trange

MINMAX = np.array([8,60])
MIN = np.array([-4,-30])

def rescale(u):
    u = (u+1)/2
    return u*MINMAX+MIN

def ood_controller():
    data = RACCARDataset(**args, pin_memory=True)
    data.setup()

    model_ref_checkpoint = torch.load(args['reference_model_path'])
    model = GaussianBC(36,64,1,512,2)
    model_ref = GaussianTrainer(model,args)
    model_ref.load_state_dict(model_ref_checkpoint['state_dict'])

    dynamic_checkpoint = torch.load(args['dynamic_model_path'])
    controller = InDCBFController(1,2,None,args['device'],latent_dim=args['latent_dim'],ode_hidden_dim=args['ode_hidden_dim'])
    # controller = InDCBFController(1,2,None,args['device'],latent_dim=args['latent_dim'],ode_hidden_dim=args['ode_hidden_dim'])
    controller = InDCBFTrainer(controller,**args)
    controller.load_state_dict(dynamic_checkpoint['state_dict'])
    controller.model.model_ref = model_ref
    controller = controller.to(args['device'])

    train_dataloader, test_dataloader = data.train_dataloader(), data.test_dataloader()
    params = {}
    x = torch.rand(16,args['latent_dim']).to(args['device'])
    layer_output_dict = controller.model.ode.get_layer_output(x)
    for idx, layer_name in enumerate(layer_output_dict.keys()):
        params[layer_name] = {}
        params[layer_name]['M'] = args['M'][idx]
        params[layer_name]['O'] = args['O'][idx]
        params[layer_name]['sig_alpha'] = args['sig_alpha'][idx]
    controller.model.init_nac_estimator(params)
    controller.model.save_neural_states(train_dataloader)

    controller.eval()
    return controller

def bc_controller():
    model_ref_checkpoint = torch.load(args['reference_model_path'])
    model = BCDensityFilter(36,64,1,512,2)
    model_ref = GaussianTrainer(model,args)
    model_ref.load_state_dict(model_ref_checkpoint['state_dict'])
    controller = model_ref.model.to(args['device'])
    controller.eval()
    return controller

def ref_controller():
    model_ref_checkpoint = torch.load(args['reference_model_path'])
    model = ReferenceController(36,64,1,512,2)
    model_ref = GaussianTrainer(model,args)
    model_ref.load_state_dict(model_ref_checkpoint['state_dict'])
    controller = model_ref.model.to(args['device'])
    controller.eval()
    return controller

def evaluation(env,args,model="ood"):
    u = torch.Tensor([0,0.]).to(args['device'])
    if model == "ood":
        controller = ood_controller()
        x = torch.zeros(1,args['latent_dim']).to(args['device'])
        u = torch.zeros(1,2).to(args['device'])
    elif model == "bc":
        controller = bc_controller()
    elif model == "ref":
        controller = ref_controller()
    model_ref = ref_controller()
    img = env.reset(pos=[20.0, -17., 0.25], hpr=[60*(np.random.rand()*2-1),0,3.14])
    img = img.reshape(img.shape[2],1,img.shape[1],img.shape[0])/255
    i = torch.Tensor(img).to(args['device'])
    interv = 0
    for frame in range(200):
        u_ref = model_ref.generate(i,u[0])
        if model == "ood":
            u_next = controller.model(i,u,x,
                                    dt=0.25,
                                    threshold=args['threshold'],
                                    sample_size=args['sample_size'],
                                    test_size=args['test_size'],)
            x = controller.model.vae(i,x,u)
        elif model == "bc":
            u_next = controller.generate(i,u, threshold = 0.95)
        elif model == "ref":
            u_next = controller.generate(i,u)
        action = u_next.view(-1).cpu().detach().numpy()
        action = rescale(action)
        action[0] = 2
        img, reward, done, info = env.step(action)
        if done:
            print("fail")
            return 0, interv/(frame+1)
        img = img.reshape(img.shape[2],1,img.shape[1],img.shape[0])/255
        u = u_next
        i = torch.Tensor(img).to(args['device'])
        interv += torch.abs(u_next-u_ref).cpu().detach().sum().numpy()
    return 1, interv/(frame+1)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--dynamic_model_path', '-dmp', default='./logs/DynamicLearning/version_14/checkpoints/best.ckpt')
    parser.add_argument('--dynamic_config_path', '-dcp', default='./logs/DynamicLearning/version_14/hparams.yaml')
    parser.add_argument('--reference_model_path', '-rmp', default='./logs/GaussianBC/version_14/checkpoints/best.ckpt')
    parser.add_argument('--M', '-m', nargs='+', type=int, default=[50,50,50,50])
    parser.add_argument('--O', '-o', nargs='+', type=int, default=[10,10,10,10])
    parser.add_argument('--sig_alpha', '-sa', nargs='+', type=float, default=[1,1,1,1])
    parser.add_argument('--threshold', '-th', type=float, default=0.95)
    parser.add_argument('--sample_size', '-ss', type=int, default=200)
    parser.add_argument('--test_size', '-ts', type=int, default=1)
    args = {k:v for (k,v) in parser.parse_args()._get_kwargs()}
    with open(args['dynamic_config_path'], 'r') as file:
        config = yaml.safe_load(file)
    for k,v in config.items():
        args[k] = v
    params = {'visualize': False, 'run_as_task': False, 'do_back_up': False, 'hfov': 120}
    env = SquareEnv(params)
    model = "ood"
    results = []
    interv_results = []
    for seed in trange(10):
        success = 0
        intervention = 0
        for _ in range(20):
            flag, interv = evaluation(env,args,model)
            success += flag
            intervention += interv
        print(success/20)
        print(intervention/20)
        results.append(success/20)
        interv_results.append(intervention/20)
    results = np.array(results)
    interv_results = np.array(interv_results)
    np.savetxt(f"./logs/Inference/{model}_success.txt",results)
    np.savetxt(f"./logs/Inference/{model}_intervention.txt",interv_results)
    print(results.mean(),results.std())
    print(interv_results.mean(),interv_results.std())
