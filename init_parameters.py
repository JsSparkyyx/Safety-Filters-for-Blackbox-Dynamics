import argparse
import torch

def init_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', '-b', type=str, choices=['vit','clip','resnet','vc1'], default="vit")
    parser.add_argument('--method',  '-m', type=str, choices=['iDBF','SABLAS','DH'], default="iDBF")
    parser.add_argument('--groundtruth', action="store_true")

    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--latent_dim', type=int, default=16)

    parser.add_argument('--w_barrier', type=float, default=1)
    parser.add_argument('--w_latent', type=float, default=1)
    parser.add_argument('--w_dyn', type=float, default=1)

    # iDBF
    parser.add_argument('--w_safe', type=float, default=1)
    parser.add_argument('--w_unsafe', type=float, default=1)
    parser.add_argument('--w_grad', type=float, default=1)

    parser.add_argument('--eps_safe', type=float, default=0.1)
    parser.add_argument('--eps_unsafe', type=float, default=0.1)
    parser.add_argument('--eps_ascent', type=float, default=0.1)
    
    # DH
    parser.add_argument('--gamma_pos', default=1)
    parser.add_argument('--gamma_neg', default=5)

    parser.add_argument('--dt',  '-dt', type=float, default=0.1)
    parser.add_argument('--rtol',  '-rtol', type=float, default=5e-6)
    parser.add_argument('--window_size', type=int, default=2)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_batch_size', type=int, default=48)
    parser.add_argument('--val_batch_size', type=int, default=24)
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=16)

    parser.add_argument('--train_barrier', default=True)
    parser.add_argument('--with_dynamic', default=True)
    parser.add_argument('--with_nonzero', default=True)
    
    parser.add_argument('--save_path', type=str, default="/storage1/sibai/Active/yuxuan/tf-logs")
    parser.add_argument('--dynamic_path', type=str, default=None)
    parser.add_argument('--name', type=str, default='iDBF')
    args = parser.parse_args()._get_kwargs()
    args = {k:v for (k,v) in args}
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    args['split_trajectory'] = True
    args['name'] = args['method']
    m2n = {"vit":"google/vit-base-patch16-224","clip":"openai/clip-vit-base-patch16","vc1":"vc1","resnet":"resnet50"}
    args['name'] += f"_{args['backbone']}_{args['seed']}"
    args['backbone'] = m2n[args['backbone']]
    if args['method'] == 'SABLAS':
        args['eps_safe'] = 0.1
        args['eps_unsafe'] = 0.1
        args['eps_ascent'] = 0.0001
        args['w_grad'] = 0.5
    return args