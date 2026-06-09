import torch
import os
import time
import pandas as pd
import math

from tqdm.auto import trange

def state2hash(states, hash_vec, batch_size=2**14, device=torch.device("cpu"), verbose=False):
    """Convert states to hashes."""
    num_batches = (states.size(0) + batch_size - 1) // batch_size
    result = torch.empty(states.size(0), dtype=torch.int64)

    if verbose:
        batch_range = trange(num_batches, desc="Hashing states")
    else:
        batch_range = range(num_batches)
    
    for i in batch_range:
        batch = states[i * batch_size:(i + 1) * batch_size].to(torch.int64).to(device)
        batch_hash = torch.sum(hash_vec * batch, dim=1).cpu()
        result[i * batch_size:(i + 1) * batch_size] = batch_hash
    return result.to(device)

class Generator:
    def __init__(self, num_millions, name="train", K_min=1, K_max=55, 
                 all_moves=None, inverse_moves=None, V0=None, device=torch.device("cpu")):
        self.device = device
        self.num_millions = num_millions
        self.epoch = 0
        self.id = int(time.time())
        self.log_dir = "logs"
        self.weights_dir = "weights"
        self.name = name
        self.K_min = K_min
        self.K_max = K_max
        self.walkers_num = num_millions * 1_000_000 // (K_max - K_min + 1)
        self.all_moves = all_moves.to(self.device)
        self.n_gens = all_moves.size(0)
        self.state_size = all_moves.size(1)
        self.inverse_moves = inverse_moves
        self.V0 = V0.to(self.device)
        self.batch_size = 10000
        self.hash_vec = torch.randint(0, int(1e15), (self.state_size,), device=self.device, dtype=torch.int64)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)

    def do_random_step(self, states, last_moves, states_hashed):
        """Perform a random step while avoiding inverse moves."""
        possible_moves = torch.ones((states.size(0), self.n_gens), dtype=torch.bool, device=self.device)
        if last_moves.sum() >= 0: 
            possible_moves[torch.arange(states.size(0)), self.inverse_moves[last_moves]] = False
        next_moves = torch.multinomial(possible_moves.float(), 1).squeeze()
        new_states = torch.gather(states, 1, self.all_moves[next_moves])
        new_states_hashed = state2hash(new_states, self.hash_vec, self.batch_size, device=self.device)
        bad_states = torch.isin(new_states_hashed, states_hashed)
        bad_states_exists = bad_states.any().item()
        while bad_states_exists:
            possible_moves[torch.arange(states.size(0)), self.inverse_moves[last_moves] * (1 - bad_states.int()) + next_moves * bad_states.int()] = False
            if not possible_moves.any(dim=1).all().item():
                return new_states, next_moves, new_states_hashed, True
            next_moves = torch.multinomial(possible_moves.float(), 1).squeeze() * bad_states.int() + next_moves * (1 - bad_states.int())
            new_states = torch.gather(states, 1, self.all_moves[next_moves])
            new_states_hashed = state2hash(new_states, self.hash_vec, self.batch_size, device=self.device)
            bad_states = torch.isin(new_states_hashed, states_hashed)
            bad_states_exists = bad_states.any().item()

        return new_states.cpu(), next_moves.cpu(), new_states_hashed, False

    def generate_random_walks(self, k=1000, K_min=1, K_max=30):
        """Random walks from K_min to K_max steps with k walkers."""
        total = k * (K_max - K_min + 1)
        Y = torch.arange(K_min, K_max + 1).repeat_interleave(k)
        states = self.V0.to('cpu').repeat(total, 1)
        states_hashed = state2hash(states, self.hash_vec, self.batch_size, device=self.device, verbose=True)
        last_moves = torch.full((total,), -1, dtype=torch.int64)
        regeneration = False
        for t in trange(K_max, desc="Generating random walks"):
            cutoff = 0 if t < K_min else k * (t - K_min + 1)
            if cutoff < total:
                j = 0
                # Note: the loop below is not just for batching; it also allows to regenerate only the remaining part of the batch when some walkers get stuck.
                for i in trange(cutoff, cutoff + k, self.batch_size, desc=f"Step {t+1}"):
                    end1 = min(i + self.batch_size, cutoff + k, total)
                    start1 = i
                    if cutoff == 0:
                        end2 = end1
                        start2 = start1
                    else:
                        end2 = min(i - k + self.batch_size, cutoff)
                        start2 = i - k
                    states[start1:end1], last_moves[start1:end1], states_hashed[start1:end1], regeneration = self.do_random_step(states[start2:end2].to(self.device), last_moves[start2:end2].to(self.device), states_hashed[:cutoff])
                    if regeneration:
                        start2 = j
                        end2 = min(j + self.batch_size, cutoff + k, total, j + end1 - start1)
                        states[start1:end1], last_moves[start1:end1], states_hashed[start1:end1] = states[start2:end2], last_moves[start2:end2], states_hashed[start2:end2]
                    j = i
            else:
                break
        if self.classification:
            Y = self.inverse_moves[last_moves]
        perm = torch.randperm(total)
        states = states[perm].cpu()
        Y = Y[perm].cpu()
        
        return states, Y
    
    def generate_dataset(self):
        states, Y, _, _ = self.generate_random_walks(k=self.walkers_num, K_min=self.K_min, K_max=self.K_max)
        os.makedirs(self.name, exist_ok=True)
        torch.save((states.cpu(), Y.cpu()), os.path.join(self.name, f"train.pt"))
        