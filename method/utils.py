import torch
import numpy as np

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
    """
    This implementation follows https://github.com/bierone/ood_coverage
    """
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