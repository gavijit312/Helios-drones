from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import yaml

from src.completion.completion_net import PointCompletionAutoEncoder
from training.dataset_loaders.point_dataset import BuildingOcclusionDataset


def chamfer_distance_approx(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """
    Symmetric Chamfer Distance between predicted surface points and ground-truth shape.
    p1, p2 shape: (B, 3, N)
    """
    # Reshape to (B, N, 3)
    p1 = p1.permute(0, 2, 1)
    p2 = p2.permute(0, 2, 1)

    # Pairwise squared distances: (B, N, N)
    dist = torch.cdist(p1, p2, p=2.0) ** 2

    # Nearest neighbor distances
    min_dist_p1_to_p2 = torch.min(dist, dim=2)[0]
    min_dist_p2_to_p1 = torch.min(dist, dim=1)[0]

    return torch.mean(min_dist_p1_to_p2) + torch.mean(min_dist_p2_to_p1)


def train_completion():
    with open("configs/completion_train.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Geometry Completion] Using compute device: {device}")

    dataset = BuildingOcclusionDataset(
        data_dir=cfg["data"]["crop_dir"], num_points=cfg["data"]["num_points"]
    )

    n_train = int(len(dataset) * cfg["data"]["train_split"])
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False
    )

    print(
        f"Dataset loaded: {len(dataset)} structural models (Train: {n_train}, Val: {n_val})"
    )

    model = PointCompletionAutoEncoder(
        num_points=cfg["data"]["num_points"]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["training"]["lr"])
    )

    save_path = Path(cfg["training"]["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    epochs = cfg["training"]["epochs"]
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            partial = batch["partial"].to(device)
            complete = batch["complete"].to(device)

            optimizer.zero_grad()
            reconstructed = model(partial)
            loss = chamfer_distance_approx(reconstructed, complete)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(1, len(train_loader))

        # Evaluation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                partial = batch["partial"].to(device)
                complete = batch["complete"].to(device)
                reconstructed = model(partial)
                val_loss += chamfer_distance_approx(
                    reconstructed, complete
                ).item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        print(
            f"Epoch [{epoch:02d}/{epochs}] - Train CD: {avg_train_loss:.5f} | Val CD: {avg_val_loss:.5f}"
        )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), str(save_path))
            print(f"  -> Saved updated completion weights to {save_path}")

    print("[Complete] Occlusion recovery network trained.")


if __name__ == "__main__":
    train_completion()