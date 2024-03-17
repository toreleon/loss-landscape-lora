from typing import List, Tuple, Optional
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from datasets import Dataset
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from transformers import Trainer, TrainingArguments
from sklearn.metrics import f1_score
from local_dataset_utilities import IMDBDataset


def perturb_model(
    model: nn.Module, direction: List[torch.Tensor], alpha: float
) -> None:
    """
    Perturb the model parameters in the given direction by alpha.
    """
    with torch.no_grad():
        for p, d in zip(
            filter(lambda p: p.requires_grad, model.parameters()), direction
        ):
            p.add_(alpha * d)


def get_random_direction(
    model: nn.Module, filter_normalize: bool = True
) -> List[torch.Tensor]:
    """
    Get a random direction for each parameter in the model.
    """
    directions = [
        torch.randn(p.size()).to(p.device)
        for p in model.parameters()
        if p.requires_grad
    ]

    if filter_normalize:
        for p, d in zip(model.parameters(), directions):
            d.mul_(p.norm() / (d.norm() + 1e-10))
            # Print the size of d and p

    return directions

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'f1': f1_score(labels, predictions, average='binary')}

def compute_metrics_and_loss_over_dimensions_with_trainer(
    model: nn.Module,
    dataset: Dataset,
    directions: List[torch.Tensor],
    alphas: List[float],
    dimensions: int = 1,
    device: str = "cuda",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Setup the training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=64,
        no_cuda=(device != "cuda"),
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset,
        compute_metrics=compute_metrics if dimensions == 1 else None,
    )

    # Prepare tensors to store the results
    losses = torch.zeros(len(alphas), len(alphas) if dimensions > 1 else 1)
    f1_scores = torch.zeros(len(alphas)) if dimensions == 1 else None
    if dimensions == 1:
        pbar = tqdm(total=len(alphas), desc="Evaluating 1D Loss Landscape")
    else:
        pbar = tqdm(total=len(alphas) ** 2, desc="Evaluating 2D Loss Landscape")
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate((alphas if dimensions > 1 else [None])):
            # Perturb the model based on the dimensionality
            perturb_model(model, directions[0], alpha)
            if dimensions > 1:
                perturb_model(model, directions[1], beta)

            # Evaluate the model
            eval_result = trainer.evaluate()

            # Store the loss
            loss = eval_result["eval_loss"]
            if dimensions == 1:
                losses[i] = loss
                # Store the F1 score if computing for 1D
                f1_scores[i] = eval_result["eval_f1"]
            else:
                losses[i, j] = loss

            # Reset the model to its original state
            perturb_model(model, directions[0], -alpha)
            if dimensions > 1:
                perturb_model(model, directions[1], -beta)
            pbar.update(1)
    # Return the computed metrics
    return losses, f1_scores


def visualize_loss_landscape(
    run_name: str,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    directions: List[torch.Tensor],
    alphas: List[float],
) -> None:
    """
    Visualize the 1D, 2D, and 3D loss landscape, saving each figure separately.
    """
    # 1D Loss Landscape with F1
    fig, ax1 = plt.subplots(figsize=(12, 10))
    losses_1d, f1_scores_1d = compute_metrics_and_loss_over_dimensions_with_trainer(
        model, dataloader, directions, alphas, dimensions=1
    )
    ax1.plot(alphas, losses_1d.numpy(), label="Loss")
    ax1.set_xlabel("alpha")
    ax1.set_ylabel("Loss", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(alphas, f1_scores_1d.numpy(), color='tab:red', label="F1 Score")
    ax2.set_ylabel("F1 Score", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.title("1D Loss and F1 Score Landscape")
    plt.savefig(f"img/{run_name}_1D_loss_f1_landscape.png")
    plt.close(fig)

    # 2D Loss Landscape
    fig, ax = plt.subplots(figsize=(12, 10))
    losses_2d, _ = compute_metrics_and_loss_over_dimensions_with_trainer(
        model, dataloader, directions, alphas, dimensions=2
    )
    alphas_mesh, betas_mesh = np.meshgrid(alphas, alphas)
    contour = ax.contour(alphas_mesh, betas_mesh, losses_2d.numpy(), cmap="viridis")
    ax.clabel(contour, inline=True, fontsize=8)
    fig.colorbar(contour, ax=ax)
    ax.set_title("2D Loss Landscape")
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    plt.tight_layout()
    plt.title("2D Loss Landscape")
    plt.savefig(f"img/{run_name}_2D_loss_landscape.png")
    plt.close(fig)

    # 3D Loss Landscape
    fig = go.Figure(data=[go.Surface(z=losses_2d.numpy(), x=alphas_mesh, y=betas_mesh, colorscale='Viridis')])
    fig.update_layout(title='3D Loss Landscape', autosize=False,
                    width=700, height=700,
                    margin=dict(l=65, r=50, b=65, t=90),
                    scene=dict(
                        xaxis_title='alpha',
                        yaxis_title='beta',
                        zaxis_title='Loss'
                    ))

    # To display in an interactive window (e.g., Jupyter Notebook)
    fig.show()

    # To save the interactive plot, you can use Plotly's write_html method
    fig.write_html(f"img/{run_name}_3D_loss_landscape.html")
