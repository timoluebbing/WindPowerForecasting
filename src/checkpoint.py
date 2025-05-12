import torch

from pathlib import Path

def save_checkpoint(
    epoch: int,
    horizon: int,
    model: torch.nn.Module,
    likelihood: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_name: str,
    checkpoint_dir: str = "../checkpoints",
) -> None:
    """
    Save the model and optimizer state to a checkpoint file.

    Args:
        epoch (int): The current epoch number.
        horizon (int): The current horizon value.
        model (torch.nn.Module): The model to save.
        likelihood (torch.nn.Module): The likelihood to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        checkpoint_name (str): Name of the checkpoint file.
        checkpoint_dir (str): Directory to save the checkpoint file.
    """
    directory = Path(checkpoint_dir) / checkpoint_name
    directory.mkdir(parents=True, exist_ok=True)

    checkpoint_path = directory / f"horizon_{horizon}.pt"

    torch.save(
        {
            "epoch": epoch,
            "horizon": horizon,
            "model_state_dict": model.state_dict(),
            "likelihood_state_dict": likelihood.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to {checkpoint_path}")
    
    
def load_checkpoint(
    model: torch.nn.Module,
    likelihood: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    horizon: int,
    checkpoint_name: str,
    checkpoint_dir: str = "../checkpoints",
) -> int:
    """
    Load the model and optimizer state from a checkpoint file.

    Args:
        model (torch.nn.Module): The model to load.
        likelihood (torch.nn.Module): The likelihood to load.
        optimizer (torch.optim.Optimizer): The optimizer to load.
        horizon (int): The horizon value from the checkpoint.
        checkpoint_name (str): Name of the checkpoint file.

    Returns:
        int: The epoch number from the checkpoint.
    """
    directory = Path(checkpoint_dir) / checkpoint_name
    checkpoint_path = directory / f"horizon_{horizon}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    assert checkpoint['horizon'] == horizon, f"Horizon mismatch: {checkpoint['horizon']} vs {horizon}"
    
    model.load_state_dict(checkpoint['model_state_dict'])
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch']


