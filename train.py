import math
import numpy as np
import os
import torch
import tqdm
import wandb
from PIL import Image

import torch.optim as optim

from models import Encoder, Decoder, BernoulliVAE

wandb_run = wandb.init(
    entity="jack-morris",
    project="categorical-vae",
)

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

def create_random_image(model: BernoulliVAE, N: int, K: int, temperature: float, step: int, output_dir: str) -> Image:
    random_image = model.generate_random_image(N, K, temperature=temperature)
    pil_image = make_pil_image(random_image)
    pil_image.save(os.path.join(output_dir, f"random_step_{step}.png"))
    return pil_image

def make_pil_image(img_tensor: torch.Tensor) -> Image:
    img_tensor = img_tensor.detach().numpy().squeeze()
    random_image = (img_tensor * 255).astype(np.uint8)
    return Image.fromarray(random_image)

def main() -> None:
    batch_size = 100
    max_steps = 50_000
    num_epochs = 10
    learning_rate = 0.001
    initial_temperature = 1.0
    minimum_temperature = 0.5
    temperature_anneal_rate = 0.00003
    K = 10 # number of classes
    N = 30 # number of categorical distributions

    training_images = load_training_data()
    train_dataset = torch.utils.data.DataLoader(
        dataset=training_images,
        batch_size=batch_size,
        shuffle=True
    )

    image_shape = next(iter(train_dataset))[0][0].shape # [1, 28, 28]
    encoder = Encoder(N, K, image_shape)
    decoder = Decoder(N, K, image_shape)
    model = BernoulliVAE(encoder, decoder)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    learning_rate_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    step = 0
    temperature = initial_temperature

    # make folder for images
    output_dir = os.path.join('outputs', wandb_run.id)
    os.makedirs(output_dir)

    progress_bar = tqdm.tqdm(total=max_steps, desc='Training')
    while step < max_steps:
        for data in train_dataset: # x should be a batch of torch.Tensor spectrograms, of shape [B, F, T]
            x, labels = data
            phi, x_hat = model(x, temperature)
            # reconstruction_loss = torch.mean(torch.sum((x - x_hat) ** 2, dim=[1,2]))
            reconstruction_loss = torch.mean((x - x_hat) ** 2)
            # kl_loss = torch.mean(bernoulli_kl_divergence_canonical(phi))
            kl_loss = torch.mean(bernoulli_kl_divergence(phi))
            loss = kl_loss + reconstruction_loss
            # loss = kl_loss
            # loss = recon_loss
            progress_bar.set_description(f'Training | Recon. loss = {reconstruction_loss:.7f} / KL loss = {kl_loss:.7f}')
            loss.backward()
            optimizer.step()
            # Incrementally anneal temperature and learning rate.
            if step % 1000 == 1:
                temperature = np.maximum(initial_temperature*np.exp(-temperature_anneal_rate*step), minimum_temperature)
                learning_rate_scheduler.step() # should multiply learning rate by 0.9
            # phi.shape: torch.Size([256, 30, 10])
            random_image = create_random_image(model, N, K, temperature, step, output_dir)

            wandb.log(
                {
                    'kl_loss': kl_loss,
                    'reconstruction_loss': reconstruction_loss, 
                    'loss': loss,
                    'random_image': wandb.Image(random_image),
                    'x': wandb.Image(make_pil_image(x[0])),
                    'x_hat': wandb.Image(make_pil_image(x_hat[0])),
                    'temperature': temperature,
                    'learning_rate': learning_rate
                }, 
                step=step
            )
            step += 1
            progress_bar.update(1)
        

if __name__ == '__main__': main()