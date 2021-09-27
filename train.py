import math
import torch
import tqdm

import torch.optim as optim

from models import Encoder, Decoder, BernoulliVAE

def load_training_data():
    import torchvision
    import torchvision.transforms as transforms
    # TODO implement datasets better
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    return torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)


def bernoulli_kl_divergence(phi: torch.Tensor) -> torch.Tensor:
    """bernoulli KL loss:
        $$\ell = (1 - \Phi)\log{\dfrac{1-\Phi}{0.5}} + \Phi \log{\dfrac{\Phi}{0.5}}$$
    """
    z_eq_0_term = (1 - phi) * (torch.log(1 - phi) - math.log(0.5))
    z_eq_1_term = phi * (torch.log(phi) - math.log(0.5))
    return (z_eq_0_term + z_eq_1_term)

def bernoulli_kl_divergence_canonical(phi: torch.Tensor) -> torch.Tensor:
    import torch.distributions as dist
    q = dist.Bernoulli(phi)
    p = dist.Bernoulli(0.5)
    return dist.kl.kl_divergence(q, p)

def main() -> None:
    batch_size = 32
    K = 10 # number of classes
    N = 30 # number of categorical distributions

    training_images = load_training_data()
    train_dataset = torch.utils.data.DataLoader(
        dataset=training_images,
        batch_size=batch_size,
        shuffle=True
    )

    encoder = Encoder(N, K)
    image_shape = next(iter(train_dataset))[0][0].shape # [1, 28, 28]
    decoder = Decoder(N, K, image_shape)
    model = BernoulliVAE(encoder, decoder)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    for data in tqdm.tqdm(train_dataset, desc='Training'): # x should be a batch of torch.Tensor spectrograms, of shape [B, F, T]
        x, labels = data
        phi, x_hat = model(x)
        # recon_loss = torch.mean(torch.sum((x - x_hat) ** 2, dim=[1,2]))
        recon_loss = torch.mean((x - x_hat) ** 2)
        # kl_loss = torch.mean(bernoulli_kl_divergence_canonical(phi))
        kl_loss = torch.mean(bernoulli_kl_divergence(phi))
        loss = kl_loss + recon_loss
        # loss = kl_loss
        # loss = recon_loss
        tqdm.tqdm.write(f'Recon. loss = {recon_loss} / KL loss = {kl_loss}')
        loss.backward()
        optimizer.step()
        

if __name__ == '__main__': main()