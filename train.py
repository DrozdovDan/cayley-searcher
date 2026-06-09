import argparse
import os
import json
import torch
from pilgrim import Trainer, DatasetTrainer Pilgrim, CayleyStarHeuristicNet, StateTransformerHeuristic
from pilgrim import count_parameters, generate_inverse_moves, load_cube_data  # assuming these exist in your module

def save_model_id(model_id):
    """Append model_id to logs/model_id.txt (create file/directories if missing)."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    model_id_file = os.path.join(log_dir, "model_id.txt")
    with open(model_id_file, "a") as f:
        f.write(f"{model_id}\n")

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Train Pilgrim Model")

    # Training and architecture hyperparameters
    parser.add_argument("--epochs", type=int, default=256, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--K_min", type=int, default=1, help="Minimum K value for random walks")
    parser.add_argument("--K_max", type=int, default=30, help="Maximum K value for random walks")
    parser.add_argument("--weights", type=str, default="", help="Model weights name (without extension) in weights/ folder")
    parser.add_argument("--device_id", type=int, default=0, help="CUDA device index (ignored if CPU)")
    parser.add_argument("--use_consistent_walks", type=bool, default=False, help="Use consistent random walks for training (True or False, default False).")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the dataset directory (if not provided, datasets will be generated on the fly each epoch).")
    # Cube parameters
    parser.add_argument("--group_id", type=int, required=True, help="Group ID")
    parser.add_argument("--target_id", type=int, default=0, help="Target ID")
    # Model architecture
    parser.add_argument("--mode", type=str, default="MLP", choices=["MLP", "GNN", "SA"], help="Model architecture mode")
    # MLP parameters
    parser.add_argument("--hd1", type=int, default=1024, help="Size of the first hidden layer")
    parser.add_argument("--hd2", type=int, default=256, help="Size of the second hidden layer (0 disables it)")
    parser.add_argument("--nrd", type=int, default=4, help="Number of residual blocks (0 disables them)")
    # GNN parameters
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dimension for GNN")
    parser.add_argument("--layers", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension for GNN")
    parser.add_argument("--num_bases", type=int, default=20, help="Number of bases for R-GCN (only for GNN mode)")
    # State Transformer parameters
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers for StateTransformer")
    parser.add_argument("--d_model", type=int, default=128, help="Dimension of the model for StateTransformer")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads for StateTransformer")
    parser.add_argument("--dim_feedforward", type=int, default=512, help="Dimension of the feedforward layer for StateTransformer")


    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_id}")
    else:
        device = torch.device("cpu")
    print(f"Start training with device: {device}.")

    # Load group data
    with open(f"generators/p{int(args.group_id):03d}.json", "r") as f:
        data = json.load(f)
        # Assuming JSON has keys; if not, fall back to .values() approach you used earlier
        if isinstance(data, dict) and "moves" in data and "move_names" in data:
            all_moves = data["moves"]
            move_names = data["move_names"]
        else:
            all_moves, move_names = data.values()
        all_moves = torch.tensor(all_moves, dtype=torch.int64, device=device)

    V0 = torch.load(
        f"targets/p{int(args.group_id):03d}-t{int(args.target_id):03d}.pt",
        weights_only=True,
        map_location=device,
    )

    # Derived parameters
    n_gens = all_moves.size(0)
    state_size = all_moves.size(1)
    num_classes = torch.unique(V0).numel()

    print("Group info:")
    print(f"  # generators   {n_gens}")
    print(f"  # classes      {num_classes}")
    print(f"  state size     {state_size}")

    # Inverse moves
    inverse_moves = torch.tensor(generate_inverse_moves(move_names), dtype=torch.int64, device=device)

    # Mode for logging
    mode = args.mode

    # Model
    if mode == "MLP":
        model = Pilgrim(
            num_classes=num_classes,
            state_size=state_size,
            hd1=args.hd1,
            hd2=args.hd2,
            nrd=args.nrd,
            dropout_rate=args.dropout,
        ).to(device)
    elif mode == "GNN":
        model = CayleyStarHeuristicNet(
            state_dim=state_size,
            n_gens=n_gens,
            all_moves=all_moves,
            num_relations=n_gens,
            hidden=args.hidden,
            layers=args.layers,
            emb_dim=args.embed_dim,
            num_bases=min(args.num_bases, n_gens),
            num_symbols=V0.max().item() + 1,
        ).to(device)
    elif mode == "SA":
        model = StateTransformerHeuristic(
            state_dim=state_size,
            num_symbols=V0.max().item() + 1,
            num_layers=args.num_layers,
            d_model=args.d_model,
            nhead=args.nhead if args.nhead < 16 else 16,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Shift for negative labels if needed
    if V0.min() < 0:
        model.z_add = -V0.min().item()

    # Optional weights loading
    if len(args.weights) > 0:
        state = torch.load(f"weights/{args.weights}.pth", weights_only=True, map_location=device)
        model.load_state_dict(state)
        print(f"Weights '{args.weights}' loaded.")

    # Params count
    num_parameters = count_parameters(model)

    # Naming
    name = f"p{int(args.group_id):03d}-t{int(args.target_id):03d}"

    TrainerClass = Trainer if args.dataset_path is None else DatasetTrainer

    # Trainer
    trainer = TrainerClass(
        net=model,
        num_epochs=args.epochs,
        device=device,
        batch_size=args.batch_size,
        lr=args.lr,
        name=name,
        K_min=args.K_min,
        K_max=args.K_max,
        all_moves=all_moves,
        inverse_moves=inverse_moves,
        V0=V0,
        use_consistent_walks=args.use_consistent_walks,
        dataset_path=args.dataset_path,
    )

    # Logs
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    args_dict = vars(args).copy()
    args_dict["model_name"] = name
    args_dict["model_mode"] = mode
    args_dict["model_id"] = trainer.id
    args_dict["num_parameters"] = num_parameters

    with open(f"{log_dir}/model_{name}_{trainer.id}.json", "w") as f:
        json.dump(args_dict, f, indent=4)

    save_model_id(trainer.id)

    # Info
    print("Model info:")
    print(f"  mode          {mode}")
    print(f"  name          {name}")
    print(f"  id            {trainer.id}")
    print(f"  # parameters  {num_parameters:_}")

    # Train
    if args.dataset_path is not None:
        trainer.run(args.dataset_path)
    else:
        trainer.run()

if __name__ == "__main__":
    main()
