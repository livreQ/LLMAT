#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
Extract fixed molecular representations from pretrained GNNs.

Supported models:
  - GIN (Hu et al., ICLR 2020): 5-layer GIN pretrained on ~2M ZINC15 molecules
  - SchNet (Schütt et al., 2017): continuous-filter CNN pretrained on QM9

Usage:
    python benchmarks/MAT/extract_gnn_features.py --model gin --problem photoswitch
    python benchmarks/MAT/extract_gnn_features.py --model schnet --problem redox-mer
    python benchmarks/MAT/extract_gnn_features.py --model gin --problem all
"""
import os
import argparse
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool

from src.utils import helpers


# ──────────────────────────────────────────────────────────────────────
# GIN architecture matching Hu et al. (ICLR 2020) pretrain-gnns exactly
# Key names match the checkpoint: x_embedding{1,2}, gnns.{i}, batch_norms.{i}
# ──────────────────────────────────────────────────────────────────────

NUM_ATOM_TYPE = 120
NUM_CHIRALITY_TAG = 3
NUM_BOND_TYPE = 6
NUM_BOND_DIRECTION = 3

PRETRAINED_URLS = {
    "gin_supervised_contextpred": (
        "https://github.com/snap-stanford/pretrain-gnns/"
        "raw/master/chem/model_gin/supervised_contextpred.pth"
    ),
    "gin_supervised_infomax": (
        "https://github.com/snap-stanford/pretrain-gnns/"
        "raw/master/chem/model_gin/supervised_infomax.pth"
    ),
    "gin_supervised_edgepred": (
        "https://github.com/snap-stanford/pretrain-gnns/"
        "raw/master/chem/model_gin/supervised_edgepred.pth"
    ),
    "gin_supervised_masking": (
        "https://github.com/snap-stanford/pretrain-gnns/"
        "raw/master/chem/model_gin/supervised_masking.pth"
    ),
}


class GINConvHu(MessagePassing):
    """GINConv matching the Hu et al. pretrain-gnns implementation.

    Edge features are embedded and added to neighbor messages.
    """

    def __init__(self, emb_dim):
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight)
        nn.init.xavier_uniform_(self.edge_embedding2.weight)

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 2, dtype=torch.long,
                                     device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)

        edge_emb = (self.edge_embedding1(edge_attr[:, 0])
                     + self.edge_embedding2(edge_attr[:, 1]))
        return self.propagate(edge_index, x=x, edge_attr=edge_emb)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINPretrained(nn.Module):
    """5-layer GIN matching Hu et al. (ICLR 2020) architecture exactly.

    After loading pretrained weights, we use global mean pooling over
    node representations to get a 300-dim graph embedding.
    """

    def __init__(self, num_layer=5, emb_dim=300, drop_ratio=0.5):
        super().__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio

        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAG, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight)
        nn.init.xavier_uniform_(self.x_embedding2.weight)

        self.gnns = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layer):
            self.gnns.append(GINConvHu(emb_dim))
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x = batched_data.x
        edge_index = batched_data.edge_index
        edge_attr = batched_data.edge_attr

        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [h]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        node_repr = h_list[-1]
        graph_repr = global_mean_pool(node_repr, batched_data.batch)
        return graph_repr

    @classmethod
    def from_pretrained(cls, name="gin_supervised_contextpred",
                        cache_dir="benchmarks/MAT/data/pretrained_models"):
        os.makedirs(cache_dir, exist_ok=True)
        weight_path = os.path.join(cache_dir, f"{name}.pth")

        if not os.path.exists(weight_path):
            url = PRETRAINED_URLS[name]
            print(f"Downloading pretrained GIN from:\n  {url}")
            torch.hub.download_url_to_file(url, weight_path)

        model = cls()
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)

        # Remove classification head keys if present
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith("graph_pred_linear")}

        model.load_state_dict(state_dict, strict=True)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Loaded pretrained GIN: {name} ({n_params:,} params, {model.emb_dim}-dim)")
        return model


# ──────────────────────────────────────────────────────────────────────
# SchNet (Schütt et al., 2017) — pretrained on QM9 via PyG
# Weights from: http://www.quantum-machine.org/datasets/trained_schnet_models.zip
# Converted from schnetpack v1 format → PyG format (cached after first run)
# ──────────────────────────────────────────────────────────────────────

SCHNET_DOWNLOAD_URL = "http://www.quantum-machine.org/datasets/trained_schnet_models.zip"
SCHNET_TARGETS = {
    "energy_U0": 7, "energy_U": 8, "enthalpy_H": 9, "free_energy": 10,
    "homo": 2, "lumo": 3, "gap": 4, "zpve": 6,
}


def _convert_schnet_weights(cache_dir, target_name="qm9_energy_U0"):
    """Convert schnetpack v1 checkpoint → PyG SchNet state dict (one-time)."""
    from torch_geometric.nn.models import SchNet

    spk_dir = os.path.join(cache_dir, "trained_schnet_models")
    if not os.path.exists(spk_dir):
        zip_path = os.path.join(cache_dir, "trained_schnet_models.zip")
        print(f"Downloading SchNet weights from:\n  {SCHNET_DOWNLOAD_URL}")
        urllib.request.urlretrieve(SCHNET_DOWNLOAD_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(cache_dir)
        os.unlink(zip_path)

    spk_path = os.path.join(spk_dir, target_name, "best_model")
    spk_model = torch.load(spk_path, map_location="cpu", weights_only=False)

    model = SchNet(hidden_channels=128, num_filters=128,
                   num_interactions=6, num_gaussians=50, cutoff=10.0)

    model.embedding.weight.data.copy_(spk_model.representation.embedding.weight.data)

    for spk_int, pyg_int in zip(spk_model.representation.interactions,
                                model.interactions):
        pyg_int.mlp[0].weight.data.copy_(spk_int.filter_network[0].weight.data)
        pyg_int.mlp[0].bias.data.copy_(spk_int.filter_network[0].bias.data)
        pyg_int.mlp[2].weight.data.copy_(spk_int.filter_network[1].weight.data)
        pyg_int.mlp[2].bias.data.copy_(spk_int.filter_network[1].bias.data)
        pyg_int.lin.weight.data.copy_(spk_int.dense.weight.data)
        pyg_int.lin.bias.data.copy_(spk_int.dense.bias.data)
        pyg_int.conv.lin1.weight.data.copy_(spk_int.cfconv.in2f.weight.data)
        pyg_int.conv.lin2.weight.data.copy_(spk_int.cfconv.f2out.weight.data)
        pyg_int.conv.lin2.bias.data.copy_(spk_int.cfconv.f2out.bias.data)

    model.lin1.weight.data.copy_(
        spk_model.output_modules[0].out_net[1].out_net[0].weight.data)
    model.lin1.bias.data.copy_(
        spk_model.output_modules[0].out_net[1].out_net[0].bias.data)
    model.lin2.weight.data.copy_(
        spk_model.output_modules[0].out_net[1].out_net[1].weight.data)
    model.lin2.bias.data.copy_(
        spk_model.output_modules[0].out_net[1].out_net[1].bias.data)

    pyg_path = os.path.join(cache_dir, f"schnet_{target_name}_pyg.pth")
    torch.save(model.state_dict(), pyg_path)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Converted SchNet: {target_name} → {pyg_path} ({n_params:,} params, 128-dim)")
    return model


def load_pretrained_schnet(target_name="qm9_energy_U0",
                           cache_dir="benchmarks/MAT/data/pretrained_models"):
    from torch_geometric.nn.models import SchNet

    os.makedirs(cache_dir, exist_ok=True)
    pyg_path = os.path.join(cache_dir, f"schnet_{target_name}_pyg.pth")

    if os.path.exists(pyg_path):
        model = SchNet(hidden_channels=128, num_filters=128,
                       num_interactions=6, num_gaussians=50, cutoff=10.0)
        model.load_state_dict(torch.load(pyg_path, map_location="cpu",
                                         weights_only=True))
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Loaded pretrained SchNet: {target_name} ({n_params:,} params, 128-dim)")
        return model

    return _convert_schnet_weights(cache_dir, target_name)


def _radius_graph_simple(pos, batch, cutoff, max_num_neighbors=32):
    """Compute radius graph without pyg-lib dependency."""
    edge_src, edge_dst = [], []
    for b in batch.unique():
        mask = (batch == b)
        idx = mask.nonzero(as_tuple=True)[0]
        p = pos[idx]
        dist = torch.cdist(p, p)
        row, col = (dist < cutoff).nonzero(as_tuple=True)
        row, col = row[row != col], col[row != col]
        edge_src.append(idx[row])
        edge_dst.append(idx[col])
    edge_index = torch.stack([torch.cat(edge_src), torch.cat(edge_dst)])
    return edge_index


def schnet_embeddings(model, z, pos, batch):
    """Extract 128-dim graph embeddings from SchNet interaction layers."""
    h = model.embedding(z)
    edge_index = _radius_graph_simple(pos, batch, model.cutoff)
    row, col = edge_index
    edge_weight = (pos[row] - pos[col]).norm(dim=-1)
    edge_attr = model.distance_expansion(edge_weight)
    for interaction in model.interactions:
        h = h + interaction(h, edge_index, edge_weight, edge_attr)
    return global_mean_pool(h, batch)


def smiles_to_3d_data(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) == -1:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) == -1:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    pos = np.array(conf.GetPositions(), dtype=np.float32)
    z = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    return Data(
        z=torch.tensor(z, dtype=torch.long),
        pos=torch.tensor(pos, dtype=torch.float),
        num_nodes=len(z),
    )


# ──────────────────────────────────────────────────────────────────────
# SMILES → PyG Data (using rdkit, matching Hu et al. featurization)
# ──────────────────────────────────────────────────────────────────────

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]


def smiles_to_data(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features: [atom_type, chirality_tag]
    atom_feats = []
    for atom in mol.GetAtoms():
        atom_type = atom.GetAtomicNum()
        chirality = atom.GetChiralTag()
        atom_type_idx = ATOM_LIST.index(atom_type) if atom_type in ATOM_LIST else 0
        chirality_idx = CHIRALITY_LIST.index(chirality) if chirality in CHIRALITY_LIST else 0
        atom_feats.append([atom_type_idx, chirality_idx])

    # Edge features: [bond_type, bond_direction]
    edges_src, edges_dst, edge_feats = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        bd = bond.GetBondDir()
        bt_idx = BOND_LIST.index(bt) if bt in BOND_LIST else 0
        bd_idx = BONDDIR_LIST.index(bd) if bd in BONDDIR_LIST else 0
        edges_src += [i, j]
        edges_dst += [j, i]
        edge_feats += [[bt_idx, bd_idx], [bt_idx, bd_idx]]

    if len(edges_src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.long)
    else:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.long)

    return Data(
        x=torch.tensor(atom_feats, dtype=torch.long),
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(atom_feats),
    )


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

DATASET_CONFIG = {
    "redox-mer":   ("redox_mer_with_iupac.csv.gz", "SMILES", "Ered", False),
    "solvation":   ("redox_mer_with_iupac.csv.gz",  "SMILES", "Gsol", False),
    "kinase":      ("enamine10k.csv.gz",            "SMILES", "score", False),
    "laser":       ("laser_multi10k.csv.gz",        "SMILES",
                    "Fluorescence Oscillator Strength", True),
    "pce":         ("photovoltaics_pce10k.csv.gz",  "SMILES", "pce", True),
    "photoswitch": ("photoswitches.csv.gz",         "SMILES",
                    "Pi-Pi* Transition Wavelength", True),
}


def extract_gin_features(problem, pretrained_name, device, batch_size=64):
    csv_file, smiles_col, obj_col, maximization = DATASET_CONFIG[problem]
    data_path = "benchmarks/MAT/data/"
    dataset = pd.read_csv(data_path + csv_file)
    smiles_list = dataset[smiles_col].tolist()
    target_vals = dataset[obj_col].to_numpy()

    print(f"\nDataset: {problem}, {len(smiles_list)} molecules")

    print("Converting SMILES to graphs...")
    pyg_list = []
    valid_idx = []
    for i, smi in enumerate(tqdm.tqdm(smiles_list)):
        data = smiles_to_data(smi)
        if data is not None:
            pyg_list.append(data)
            valid_idx.append(i)
        else:
            print(f"  WARNING: failed SMILES at index {i}: {smi}")

    print(f"  {len(pyg_list)}/{len(smiles_list)} molecules converted")

    model = GINPretrained.from_pretrained(pretrained_name)
    model = model.to(device)
    model.eval()

    print("Extracting embeddings...")
    features = []
    targets_list = []

    for start in tqdm.tqdm(range(0, len(pyg_list), batch_size)):
        batch_data = pyg_list[start:start + batch_size]
        batch = Batch.from_data_list(batch_data).to(device)

        with torch.no_grad():
            embs = model(batch)

        for j, emb in enumerate(embs.cpu()):
            features.append(emb)
            orig_idx = valid_idx[start + j]
            t = helpers.y_transform(
                torch.tensor([target_vals[orig_idx]]), maximization
            )
            targets_list.append(t.float())

    print(f"  Feature dim: {features[0].shape[0]}")

    feature_name = "gin"
    cache_path = f"benchmarks/MAT/data/features/{problem}/"
    os.makedirs(cache_path, exist_ok=True)
    torch.save(features, cache_path + f"{feature_name}_feats.bin")
    torch.save(targets_list, cache_path + f"{feature_name}_targets.bin")
    print(f"  Saved to {cache_path}{feature_name}_feats.bin / _targets.bin")


def extract_schnet_features(problem, schnet_target, device, batch_size=64):
    csv_file, smiles_col, obj_col, maximization = DATASET_CONFIG[problem]
    data_path = "benchmarks/MAT/data/"
    dataset = pd.read_csv(data_path + csv_file)
    smiles_list = dataset[smiles_col].tolist()
    target_vals = dataset[obj_col].to_numpy()

    print(f"\nDataset: {problem}, {len(smiles_list)} molecules")

    print("Generating 3D conformers...")
    pyg_list = []
    valid_idx = []
    for i, smi in enumerate(tqdm.tqdm(smiles_list)):
        data = smiles_to_3d_data(smi)
        if data is not None:
            pyg_list.append(data)
            valid_idx.append(i)
        else:
            print(f"  WARNING: 3D generation failed at index {i}: {smi}")

    print(f"  {len(pyg_list)}/{len(smiles_list)} molecules converted")

    model = load_pretrained_schnet(target_name=schnet_target)
    model = model.to(device)
    model.eval()

    print("Extracting embeddings...")
    features = []
    targets_list = []

    for start in tqdm.tqdm(range(0, len(pyg_list), batch_size)):
        batch_data = pyg_list[start:start + batch_size]
        batch = Batch.from_data_list(batch_data).to(device)

        with torch.no_grad():
            embs = schnet_embeddings(model, batch.z, batch.pos, batch.batch)

        for j, emb in enumerate(embs.cpu()):
            features.append(emb)
            orig_idx = valid_idx[start + j]
            t = helpers.y_transform(
                torch.tensor([target_vals[orig_idx]]), maximization
            )
            targets_list.append(t.float())

    print(f"  Feature dim: {features[0].shape[0]}")

    feature_name = "schnet"
    cache_path = f"benchmarks/MAT/data/features/{problem}/"
    os.makedirs(cache_path, exist_ok=True)
    torch.save(features, cache_path + f"{feature_name}_feats.bin")
    torch.save(targets_list, cache_path + f"{feature_name}_targets.bin")
    print(f"  Saved to {cache_path}{feature_name}_feats.bin / _targets.bin")


def main():
    parser = argparse.ArgumentParser(
        description="Extract pretrained GNN features."
    )
    parser.add_argument(
        "--model",
        choices=["gin", "schnet"],
        default="gin",
    )
    parser.add_argument(
        "--problem",
        choices=list(DATASET_CONFIG.keys()) + ["all"],
        default="photoswitch",
    )
    parser.add_argument(
        "--pretrained",
        choices=list(PRETRAINED_URLS.keys()),
        default="gin_supervised_contextpred",
        help="GIN pretrained variant (only used with --model gin)",
    )
    parser.add_argument(
        "--schnet_target",
        choices=list(SCHNET_TARGETS.keys()),
        default="energy_U0",
        help="QM9 target property for SchNet weights (only used with --model schnet)",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    problems = list(DATASET_CONFIG.keys()) if args.problem == "all" else [args.problem]
    for problem in problems:
        if args.model == "gin":
            extract_gin_features(problem, args.pretrained, device, args.batch_size)
        elif args.model == "schnet":
            schnet_target = f"qm9_{args.schnet_target}"
            extract_schnet_features(problem, schnet_target, device, args.batch_size)

    print("\nDone.")


if __name__ == "__main__":
    main()
