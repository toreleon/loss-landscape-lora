from typing import List, Any
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
from datasets import Dataset
import matplotlib.pyplot as plt

def perturb_model(model: nn.Module, direction: List[torch.Tensor], alpha: float) -> None:
    """
    Perturb the model parameters in the given direction by alpha.
    """
    with torch.no_grad():
        for p, d in zip(filter(lambda p: p.requires_grad, model.parameters()), direction):
            p.add_(alpha * d)

def get_random_direction(model: nn.Module, filter_normalize: bool = True) -> List[torch.Tensor]:
    """
    Get a random direction for each parameter in the model.
    """
    directions = [torch.randn(p.size()).to(p.device) for p in model.parameters() if p.requires_grad]

    if filter_normalize:
        for p, d in zip(model.parameters(), directions):
            d.mul_(p.norm() / (d.norm() + 1e-10))
            # Print the size of d and p

    return directions


def compute_loss_over_dimensions(model: nn.Module, criterion: nn.Module, dataloader: torch.utils.data.DataLoader, directions: List[torch.Tensor], alphas: List[float], dimensions: int = 2) -> torch.Tensor:
    """
    Compute the average loss over 1D, 2D, or 3D defined by directions and alphas using data from a DataLoader.
    """
    assert dimensions in [1, 2, 3], "Only 1D, 2D, and 3D computations are supported."
    
    # Initialize the losses tensor based on the number of dimensions
    if dimensions == 1:
        losses = torch.zeros(len(alphas))
    else:
        losses = torch.zeros(len(alphas), len(alphas))
    pbar = tqdm(total=len(alphas) ** dimensions, desc=f'Computing loss over {dimensions}D')
    
    # Nested loops to iterate through the dimensions
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate((alphas if dimensions > 1 else [None])):
            total_loss = 0
            total_samples = 0
            for batch in dataloader:
                # Load batch to device
                batch = {k: v.to("cuda") for k, v in batch.items()}
                # Unpack the batch
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                # Perturb model based on the number of dimensions
                if dimensions >= 1:
                    perturb_model(model, directions[0], alpha)
                if dimensions > 1:
                    perturb_model(model, directions[1], beta)
                
                # Compute loss
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = criterion(outputs.logits, labels)
                total_loss += loss.item() * len(labels)
                total_samples += len(labels)
                # Reset model to its original state
                if dimensions >= 1:
                    perturb_model(model, directions[0], -alpha)
                if dimensions > 1:
                    perturb_model(model, directions[1], -beta)
            avg_loss = total_loss / total_samples
            if dimensions == 1:
                losses[i] = avg_loss
            else:
                losses[i, j] = avg_loss
            pbar.update(1)
    pbar.close()
    return losses

def visualize_loss_landscape(model: nn.Module, criterion: nn.Module, dataloader: torch.utils.data.DataLoader, directions: List[torch.Tensor], alphas: List[float]) -> None:
    """
    Visualize the 1D, 2D, and 3D loss landscape, saving each figure separately.
    """
    # 1D Loss Landscape
    fig, ax = plt.subplots(figsize=(6, 4))
    losses_1d = compute_loss_over_dimensions(model, criterion, dataloader, directions, alphas, dimensions=1)
    ax.plot(alphas, losses_1d.numpy())
    ax.set_title('1D Loss Landscape')
    ax.set_xlabel('alpha')
    ax.set_ylabel('Loss')
    plt.tight_layout()
    plt.savefig('1D_loss_landscape.png')
    plt.close(fig)

    # 2D Loss Landscape
    fig, ax = plt.subplots(figsize=(6, 4))
    losses_2d = compute_loss_over_dimensions(model, criterion, dataloader, directions, alphas, dimensions=2)
    alphas_mesh, betas_mesh = np.meshgrid(alphas, alphas)
    contour = ax.contour(alphas_mesh, betas_mesh, losses_2d.numpy(), cmap='viridis')
    ax.clabel(contour, inline=True, fontsize=8)
    fig.colorbar(contour, ax=ax)
    ax.set_title('2D Loss Landscape')
    ax.set_xlabel('alpha')
    ax.set_ylabel('beta')
    plt.tight_layout()
    plt.savefig('2D_loss_landscape.png')
    plt.close(fig)

    # 3D Loss Landscape
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(alphas_mesh, betas_mesh, losses_2d, cmap='viridis', edgecolor='none')
    ax.set_title('3D Loss Landscape')
    ax.set_xlabel('alpha')
    ax.set_ylabel('beta')
    ax.set_zlabel('Loss')
    plt.tight_layout()
    plt.savefig('3D_loss_landscape.png')
    plt.close(fig)