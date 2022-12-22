import torch
import torch.nn as nn
import math

actv_dict = {
    'sigmoid' : nn.Sigmoid(),
    'softplus' : nn.Softplus(),
    'relu' : nn.ReLU(True)
}

def _xavier_init(linear):
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    nn.init.xavier_uniform_(linear.weight.data)

class Shift(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, x):
        return x+self.val

class MLP(nn.Module):
    """
    A simple MLP.
    """

    def __init__(self, 
                net_depth: int, 
                net_width: int, 
                skip_index: int, 
                input_dim: int, 
                output_dim: int,
                activation: str,
                last_activation: bool = True):
        """
          net_depth: The depth of the first part of MLP.
          net_width: The width of the first part of MLP.
          activation: The activation function.
          skip_index: Add a skip connection to the output of every N layers.
        """
        super(MLP, self).__init__()
        self.skip_index: int = skip_index  # Add a skip connection to the output of every N layers.
        
        layers = []
        for i in range(net_depth):
            if i == 0:                                  # first layer
                dim_in = input_dim
            elif (i-1) % skip_index == 0 and i > 1:     # skip conn.
                dim_in = net_width + input_dim
            else:
                dim_in = net_width

            if i == net_depth-1:                        # last layer
                dim_out = output_dim
            else:
                dim_out = net_width

            linear = nn.Linear(dim_in, dim_out)
            # _xavier_init(linear)
            
            if i == net_depth-1 and not last_activation:
                layers.append(nn.Sequential(linear))
            else:
                layers.append(nn.Sequential(linear, actv_dict[activation]))

        self.layers = nn.ModuleList(layers)
        del layers

    def forward(self, x, view_direction=None):
        """Evaluate the MLP.

        Args:
            x: torch.Tensor(float32), [batch, num_samples, feature], points.
            view_direction: torch.Tensor(float32), [batch, feature], if not None, this
            variable will be part of the input to the second part of the MLP
            concatenated with the output vector of the first part of the MLP. If
            None, only the first part of the MLP will be used with input x. In the
            original paper, this variable is the view direction.

        Returns:
            raw_rgb: torch.Tensor(float32), with a shape of
                [batch, num_samples, 3].
            raw_density: torch.Tensor(float32), with a shape of
                [batch, num_samples, 1].
        """
        num_samples = x.shape[1]
        inputs = x  # [B, N, 2*3*L]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i % self.skip_index == 0 and i > 0:
                x = torch.cat([x, inputs], dim=-1)

        return x


class PosEmbedding(nn.Module):
    def __init__(self, degree, identity=False):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super().__init__()
        self.degree = degree
        self.identity = identity
        self.funcs = [torch.sin, torch.cos]
        self.freqs = 2**torch.linspace(0, degree-1, degree)

    def get_dim(self):
        if self.identity:
            return self.degree*4 + 2
        else:
            return self.degree*4

    def forward(self, x):
        """
        Inputs:
            x: (B, C)

        Outputs:
            out: (B, 2*C*degree+C)
        """
        out = []
        if self.identity:
            out += [x]
        for freq in self.freqs:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class Concrete(nn.Module):

    """Concrete Distribution.
    Implementation of the Concrete Dropout module as described in the
    'Concrete Dropout' paper: https://arxiv.org/pdf/1705.07832
    """

    def __init__(self, tmp=0.9, u_min=0.0, u_max=1.0, eps=1e-8, t_min=0.0, gamma=1, concrete_tmp_anneal_step=4000):
        """Concrete Distribution.
        Parameters
        ----------
            tmp: float
                temperature 
            u_min: float 
                noise min value
            u_max: float
                noise max value
            eps: float
                epsilon to prevent 0 input in log function
            t_min: float
                min tmp value if using annealing
            gamma: float
                multiplier for annealing
            concrete_tmp_anneal_step: float
                update step interval
        """
        super().__init__()
        self.tmp = tmp
        self.u_min = u_min
        self.u_max = u_max
        self.eps = eps
        self.t_min = t_min
        self.gamma = gamma
        self.anneal_step = concrete_tmp_anneal_step


    def forward(self, x, step, randomized=True):
        if not randomized:
            u_noise = torch.zeros_like(x)+0.5
        else:
            u_noise = (self.u_max-self.u_min)*torch.rand_like(x) + self.u_min
        prob = torch.log(x + self.eps) + \
                torch.log(u_noise + self.eps) - \
                torch.log(1 - u_noise + self.eps)
        tmp = max(self.t_min, self.tmp*math.exp(-self.gamma*(step//self.anneal_step * self.anneal_step)))
        prob = torch.sigmoid(prob / tmp)
        return prob, tmp