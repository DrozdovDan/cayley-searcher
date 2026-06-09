import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out


class Pilgrim(nn.Module):
    def __init__(self, state_size, hd1=5000, hd2=1000, nrd=2, output_dim=1, dropout_rate=0.0, num_classes=6):
        super(Pilgrim, self).__init__()
        self.dtype = torch.float32
        self.state_size = state_size
        self.num_classes = num_classes
        self.hd1 = hd1
        self.hd2 = hd2
        self.nrd = nrd
        self.z_add = 0

        self.input_layer = nn.Linear(state_size * num_classes, hd1)
        self.bn1 = nn.BatchNorm1d(hd1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        if hd2 > 0:
            self.hidden_layer = nn.Linear(hd1, hd2)
            self.bn2 = nn.BatchNorm1d(hd2)
            hidden_dim_for_output = hd2
        else:
            self.hidden_layer = None
            self.bn2 = None
            hidden_dim_for_output = hd1

        if nrd > 0 and hd2 > 0:
            self.residual_blocks = nn.ModuleList(
                [ResidualBlock(hd2, dropout_rate) for _ in range(nrd)]
            )
        else:
            self.residual_blocks = None

        self.output_layer = nn.Linear(hidden_dim_for_output, output_dim)

    def forward(self, z):
        # One-hot encode and flatten to dense
        x = F.one_hot(z.long() + self.z_add, num_classes=self.num_classes).view(z.size(0), -1).to(self.dtype)

        # Input block
        x = self.input_layer(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Optional hidden block
        if self.hidden_layer is not None:
            x = self.hidden_layer(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.dropout(x)

        # Optional residual stack
        if self.residual_blocks is not None:
            for block in self.residual_blocks:
                x = block(x)

        # Output
        x = self.output_layer(x)
        return x.flatten()


def count_parameters(model):
    """Count the trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def batch_process(model, data, device, batch_size):
    """
    Process data through a model in batches.

    :param data: Tensor of input data
    :param model: A PyTorch model with a forward method that accepts data
    :param device: Device to perform computations (e.g., 'cuda', 'cpu')
    :param batch_size: Number of samples per batch
    :return: Concatenated tensor of model outputs
    """
    model.eval()
    model.to(device)

    outputs = torch.empty(data.size(0), dtype=torch.float16, device=device)

    # Process each batch
    for i in range(0, data.size(0), batch_size):
        batch = data[i:i + batch_size].to(device)
        with torch.no_grad():
            batch_output = model(batch).flatten()
        outputs[i:i + batch_size] = batch_output

    return outputs

class CayleyStarHeuristicNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_gens: int,
        all_moves: torch.Tensor,       # [n_gens, state_dim], long
        num_relations: int,
        hidden: int = 256,
        layers: int = 3,
        num_bases: int | None = None,
        num_symbols: int | None = None,
    ):
        super().__init__()
        assert all_moves.dtype == torch.long
        assert all_moves.shape == (n_gens, state_dim)

        self.state_dim = state_dim
        self.n_gens = n_gens
        self.num_symbols = num_symbols
        self.num_relations = num_relations

        # register permutation indices as buffer
        self.register_buffer("all_moves", all_moves)

        # precompute a single-graph star edge_index / edge_type (root=0, neighbors=1..n_gens)
        N = 1 + n_gens
        # edges: root -> i and i -> root
        src = torch.cat([torch.zeros(n_gens, dtype=torch.long),
                         torch.arange(1, N, dtype=torch.long)])
        dst = torch.cat([torch.arange(1, N, dtype=torch.long),
                         torch.zeros(n_gens, dtype=torch.long)])
        edge_index_1 = torch.stack([src, dst], dim=0)  # [2, 2*n_gens]
        edge_type_1 = torch.cat([
            torch.arange(n_gens, dtype=torch.long),
            torch.arange(n_gens, dtype=torch.long),
        ])  # [2*n_gens]

        self.register_buffer("edge_index_1", edge_index_1)
        self.register_buffer("edge_type_1", edge_type_1)

        # input feature dim depends on whether you pass float features or int-coded states
        in_channels = state_dim if num_symbols is None else state_dim * num_symbols

        self.lin_in = nn.Linear(in_channels, hidden)
        self.convs = nn.ModuleList([
            RGCNConv(hidden, hidden, num_relations=num_relations, num_bases=num_bases)
            for _ in range(layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _to_features(self, states: torch.Tensor) -> torch.Tensor:
        """
        states: [B, state_dim] (long or float)
        returns float features [B, Fin]
        """
        if states.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
            if self.num_symbols is None:
                raise ValueError("states are integer-coded but num_symbols=None")
            # one-hot per position, flatten
            x = F.one_hot(states.long(), num_classes=self.num_symbols).float()
            return x.view(states.size(0), -1)  # [B, state_dim*num_symbols]
        else:
            return states.float()  # [B, state_dim] or [B, Fin] if you already flattened

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        states: [B, state_dim] (int-coded) OR [B, Fin] (float features)
        returns: [B] float heuristics for the root states
        """
        B = states.size(0)
        device = states.device

        # 1) build successors via gather using permutations
        # torch.gather requires long index tensor. :contentReference[oaicite:2]{index=2}
        if states.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
            base = states.unsqueeze(1).expand(B, self.n_gens, self.state_dim)  # [B,G,D]
            idx = self.all_moves.unsqueeze(0).expand(B, self.n_gens, self.state_dim)  # [B,G,D]
            succ = base.gather(2, idx)  # [B,G,D] int-coded
        else:
            # float features: permutations must index the feature dimension you want to permute.
            D = states.size(1)
            base = states.unsqueeze(1).expand(B, self.n_gens, D)  # [B,G,D]
            idx = self.all_moves.unsqueeze(0).expand(B, self.n_gens, self.all_moves.size(1))  # [B,G,state_dim]
            if idx.size(2) != D:
                raise ValueError(
                    f"Float input has D={D}, but all_moves indexes length={idx.size(2)}. "
                    "Either pass int-coded states, or provide all_moves indexing the feature dim."
                )
            succ = base.gather(2, idx)  # [B,G,D] float

        # nodes per graph: [root] + [succ_1..succ_G]
        nodes = torch.cat([states.unsqueeze(1), succ], dim=1)  # [B, 1+G, ...]
        nodes_flat = nodes.view(B * (1 + self.n_gens), -1)     # [B*(1+G), ...]

        # 2) node features
        x = self._to_features(nodes_flat)  # float
        x = F.relu(self.lin_in(x))

        # 3) build batched star edges by offsetting indices
        N = 1 + self.n_gens
        E1 = self.edge_index_1.size(1)

        offsets = (torch.arange(B, device=device, dtype=torch.long) * N)  # [B]
        edge_index = (self.edge_index_1.unsqueeze(0) + offsets.view(B, 1, 1))  # [B,2,E1]
        edge_index = edge_index.permute(1, 0, 2).reshape(2, B * E1)            # [2,B*E1]

        edge_type = self.edge_type_1.repeat(B)  # [B*E1], must be long and in-range :contentReference[oaicite:3]{index=3}

        # 4) message passing
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)

        # 5) read out root nodes only
        root_idx = offsets  # root is the first node in each star
        root_x = x[root_idx]                  # [B, hidden]
        h = self.head(root_x).squeeze(-1)     # [B]
        return h

class StateTransformerHeuristic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_symbols: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        nonneg: bool = True,
        output_dim: int = 1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_symbols = num_symbols
        self.d_model = d_model
        self.nonneg = nonneg

        self.tok = nn.Embedding(num_symbols, d_model)
        self.pos = nn.Embedding(state_dim + 1, d_model)

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        states = states.long()
        B, D = states.shape

        x = self.tok(states)  # [B, D, d_model]

        pos_idx = torch.arange(1, D + 1, device=states.device)
        x = x + self.pos(pos_idx).unsqueeze(0)
        cls = self.cls.expand(B, -1, -1) + self.pos(
            torch.zeros(1, device=states.device, dtype=torch.long)
        ).view(1, 1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, D+1, d_model]
        x = self.enc(x)
        out = x[:, 0, :]       # CLS

        y = self.head(out).squeeze(-1)
        return F.softplus(y) if self.nonneg else y
