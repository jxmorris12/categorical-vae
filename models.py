from typing import Tuple

import torch
import torchvision

import torch.distributions as dist


def gumbel_distribution_sample(shape: torch.Size, eps=1e-20) -> torch.Tensor:
    """Samples from the Gumbel distribution given a tensor shape and value of epsilon.
    
    note: the \eps here is just for numerical stability. The code is basically just doing
            > -log(-log(rand(shape)))
    where rand generates random numbers on U(0, 1). 
    """
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_distribution_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Adds Gumbel noise to `logits` and applies softmax along the last dimension.
    
    Softmax is applied wrt a given temperature value. A higher temperature will make the softmax
    softer (less spiky). Lower temperature will make softmax more spiky and less soft. As
    temperature -> 0, this distribution approaches a categorical distribution.
    """
    y = logits + gumbel_distribution_sample(logits.shape)
    return torch.nn.functional.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits: torch.Tensor, temperature: float, hard=False) -> torch.Tensor:
    """
    ST-gumple-softmax
    input: [*, n_classes]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_distribution_sample(logits, temperature)    
    input_shape = y.shape
    n_classes = input_shape[-1] # TODO(jxm): check this!
    if hard:
        # Replace y with a one-hot vector, y_hard.
        _, max_indices = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, input_shape[-1])
        y_hard.scatter_(1, max_indices.view(-1, 1), 1)
        y_hard = y_hard.view(input_shape)
        # This line basically says: give y_hard the gradients of y,
        # but retain the value of y_hard.
        y_hard = (y_hard - y).detach() + y
        return y_hard.view(input_shape)
    else:
        return y.view(input_shape)

class Encoder(torch.nn.Module):
    cnn: torch.nn.Module
    N: int # number of categorical distributions
    K: int # number of classes
    def __init__(self, N: int, K: int):
        super().__init__()
        self.N = N
        self.K = K

        print(f'initializing encoder with  N={N} and K={K}')
        print(f'using MobileNetV3 encoder with {N*K} classes')
        self.cnn = torchvision.models.mobilenet_v3_small(
            pretrained=False, num_classes=N*K
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produces encoding `z` for input spectrogram `x`.
        
        Actually returns theta, the parameters of a bernoulli producing `z`.
        """
        assert len(x.shape) == 4 # x should be of shape [B, C, Y, X]
        B = x.shape[0] # store batch dimension
        x = x.repeat(1, 3, 1, 1) # add fake channels
        phi = torch.reshape(self.cnn(x), (B, self.N, self.K))
        return torch.sigmoid(phi) # TODO ensure sigmoid dim is correct


class Decoder(torch.nn.Module):
    output_shape: torch.Size
    N: int # number of categorical distributions
    K: int # number of classes
    def __init__(self, N: int, K: int, output_shape: torch.Size):
        super().__init__()
        self.N = N
        self.K = K
        self.output_shape = output_shape
        # TODO make a smarter decoder with deconv?
        self.ff = torch.nn.Linear(
            N*K,
            torch.prod(torch.tensor(output_shape))
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Produces output spectrogram `x` for input `z`."""
        assert len(z.shape) == 3 # [B, N, K]
        assert z.shape[1:] == (self.N, self.K)

        x_hat = self.ff(torch.reshape(z, (-1, self.N*self.K)))
        x_hat = torch.reshape(x_hat, [-1] + list(self.output_shape))
        return torch.sigmoid(x_hat)


class BernoulliVAE(torch.nn.Module):
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """VAE forward pass. Encoder produces phi, the parameters of a bernoulli distribution.
        Samples from bernoulli(phi) to produce a z. Passes z through encoder p(x|z) to get x_hat,
        a reconstruction of x.

        Returns:
            phi: parameters of categorical distribution that produced z
            x_hat: auto-encoder reconstruction of x
        """
        phi = self.encoder(x)
        print('phi.shape:', phi.shape)
        # TODO anneal temp during training?
        # z_given_x = torch.nn.functional.gumbel_softmax(phi, tau=1, hard=False)
        #phi = torch.stack((phi, 1-phi), dim=-1) # split into 2 classes
        z_given_x = gumbel_softmax(phi, 1.0, hard=False)
        print('z_given_x.shape', z_given_x.shape)
        x_hat = self.decoder(z_given_x)
        print('x_hat.shape:', x_hat.shape)
        return phi, x_hat

