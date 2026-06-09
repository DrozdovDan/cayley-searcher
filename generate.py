import argparse
import torch
import os
import json
from pilgrim import Trainer, Pilgrim, count_parameters, generate_inverse_moves, load_cube_data
from pilgrim.trainer import HammingTrainer
from pilgrim.dataset_generator import Generator, ManhattanGenerator

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate dataset for training Pilgrim Model")
    
    # Generator hyperparameters
    parser.add_argument("--millions", type=int, default=128, help="Number of millions of samples to generate")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size")
    parser.add_argument("--K_min", type=int, default=1, help="Minimum K value for random walks")
    parser.add_argument("--K_max", type=int, default=30, help="Maximum K value for random walks")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID")
    parser.add_argument("--name", type=str, default="dataset", help="Name for the generation session and saved files.")
    # Cube parameters
    parser.add_argument("--group_id", type=int, help="Group ID.")
    parser.add_argument("--target_id", type=int, default=0, help="Target ID.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", args.device_id)

    print(f"Start training with {device}.")

    # Load group data (moves, names, target)
    with open(f'generators/p{int(args.group_id):03d}.json', 'r') as f:
        all_moves, move_names = json.load(f).values()
        all_moves = torch.tensor(all_moves, dtype=torch.int64, device=device)
    V0 = torch.load(f"targets/p{int(args.group_id):03d}-t{int(args.target_id):03d}.pt", weights_only=True, map_location=device)

    # Derive important group parameters from the loaded data
    n_gens = all_moves.size(0)  # Number of moves
    state_size = all_moves.size(1)  # Size of the state representation
    num_classes = torch.unique(V0).size(0)
    print(f"Group info:")
    print(f"  # generators   {n_gens}")
    print(f"  # classes      {num_classes}")
    print(f"  state size     {state_size}")


    # Generate inverse moves
    inverse_moves = torch.tensor(generate_inverse_moves(move_names), dtype=torch.int64, device=device)

    generator = Generator(
        num_millions=args.millions,
        name=args.name,
        K_min=args.K_min,
        K_max=args.K_max,
        all_moves=all_moves,
        inverse_moves=inverse_moves,
        V0=V0,
        device=device,
    )
    generator.generate_dataset()

if __name__ == "__main__":
    main()