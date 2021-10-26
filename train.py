import math
import numpy as np
import os
import torch
import torchvision
import tqdm
import wandb
from PIL import Image

import torch.distributions as dist
import torch.optim as optim
import torchvision.transforms as transforms

from models import Encoder, Decoder, CategoricalVAE

use_wandb = False

if use_wandb:
    wandb_run = wandb.init(
        entity="jack-morris",
        project="categorical-vae",
    )
    training_run_id = wandb_run.id
else:
    training_run_id = 'default'

def load_training_data():
    # TODO implement datasets better
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    transform = transforms.Compose([transforms.ToTensor()])
    return torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)

def categorical_kl_divergence(phi: torch.Tensor) -> torch.Tensor:
    # phi is logits of shape [B, N, K] where B is batch, N is number of categorical distributions, K is number of classes
    B, N, K = phi.shape
    phi = phi.view(B*N, K)
    q = dist.Categorical(logits=phi)
    p = dist.Categorical(probs=torch.full((B*N, K), 1.0/K)) # uniform bunch of K-class categorical distributions
    kl = dist.kl.kl_divergence(q, p) # kl is of shape [B*N]
    return kl.view(B, N)

def create_random_image(model: CategoricalVAE, N: int, K: int, temperature: float, step: int, output_dir: str) -> Image:
    random_image = model.generate_random_image(N, K, temperature=temperature)
    pil_image = make_pil_image(random_image)
    pil_image.save(os.path.join(output_dir, f"random_step_{step}.png"))
    return pil_image

def make_pil_image(img_tensor: torch.Tensor) -> Image:
    img_tensor = img_tensor.detach().numpy().squeeze()
    random_image = (img_tensor * 255).astype(np.uint8)
    return Image.fromarray(random_image)

def main() -> None:
    # TODO rewrite to use NamedTensor?
    # baseline: https://github.com/harvardnlp/namedtensor/blob/master/examples/vae.py
    wandb_log_interval = 10
    model_save_interval = 5_000
    batch_size = 100
    max_steps = 50_000
    initial_learning_rate = 0.001
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
    model = CategoricalVAE(encoder, decoder)

    if use_wandb:
        wandb.watch(model, log_freq=wandb_log_interval)

    parameters = list(model.parameters())
    optimizer = optim.SGD(parameters, lr=initial_learning_rate, momentum=0.0)
    learning_rate_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    step = 0
    temperature = initial_temperature

    # make folder for images
    output_dir = os.path.join('outputs', training_run_id)
    os.makedirs(output_dir, exist_ok=True)

    progress_bar = tqdm.tqdm(total=max_steps, desc='Training')
    while step < max_steps:
        for data in train_dataset: # x should be a batch of torch.Tensor inputs, of shape [B, F, T]
            x, labels = data
            phi, x_hat = model(x, temperature) # phi shape: [B, N, K]; x_hat shape: [B, C, Y, X]
            reconstruction_loss = (
               torch.nn.functional.binary_cross_entropy(x_hat, x, reduction="none").sum()) / x.shape[0]
            kl_loss = torch.mean(
                torch.sum(categorical_kl_divergence(phi), dim=1)
            )
            loss = kl_loss + reconstruction_loss
            progress_bar.set_description(f'Training | Recon. loss = {reconstruction_loss:.7f} / KL loss = {kl_loss:.7f}')
            gradnorm = torch.nn.utils.clip_grad_norm_(parameters, 1)
            loss.backward()
            optimizer.step()

            # Incrementally anneal temperature and learning rate.
            if step % 1000 == 1:
                temperature = np.maximum(initial_temperature*np.exp(-temperature_anneal_rate*step), minimum_temperature)
                learning_rate_scheduler.step() # should multiply learning rate by 0.9

            if step % wandb_log_interval == 0:
                if use_wandb:
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
                            #'phi_hist': wandb.Histogram(phi.exp().detach().numpy().flatten()),
                            #'phi_sum_hist': wandb.Histogram(phi.exp().sum(axis=2).detach().numpy().flatten()),
                            #'x_hat_hist': wandb.Histogram(x_hat.detach().numpy().flatten()),
                            #'x_hist': wandb.Histogram(x.detach().numpy().flatten()),
                            'learning_rate': learning_rate_scheduler.get_lr()
                        }, 
                        step=step
                    )
            if (step+1) % model_save_interval == 0:
                torch.save(model.state_dict(), os.path.join(output_dir, f'save_{step}.pt'))

            step += 1
            progress_bar.update(1)
        

if __name__ == '__main__': main()