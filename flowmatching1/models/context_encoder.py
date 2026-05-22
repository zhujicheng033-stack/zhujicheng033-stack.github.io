"""Context encoding for multi-condition perturbation experiments."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List


class ContextEncoder(nn.Module):
    """
    Encode discrete conditions (drug, cell-line, dose) into continuous embeddings.
    
    v_θ(x, t, c) uses c to modulate behavior across conditions.
    """
    
    def __init__(
        self,
        n_drugs: int,
        n_cell_lines: int,
        embedding_dim: int = 8,
        dose_embedding_dim: int = 4,
    ):
        """
        Args:
            n_drugs: number of different drugs
            n_cell_lines: number of different cell lines
            embedding_dim: dimension of drug/cell-line embeddings
            dose_embedding_dim: dimension of dose encoding
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.dose_embedding_dim = dose_embedding_dim
        
        # Embedding layers for discrete conditions
        self.drug_embedding = nn.Embedding(n_drugs, embedding_dim)
        self.cell_line_embedding = nn.Embedding(n_cell_lines, embedding_dim)
        
        # Dose encoding: map continuous dose to embedding
        self.dose_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, dose_embedding_dim),
        )
        
        self.output_dim = 2 * embedding_dim + dose_embedding_dim
    
    def forward(
        self,
        drug_ids: torch.Tensor,
        cell_line_ids: torch.Tensor,
        doses: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Encode conditions to context vector.
        
        Args:
            drug_ids: integer IDs for drugs, shape (B,)
            cell_line_ids: integer IDs for cell lines, shape (B,)
            doses: continuous dose values, shape (B, 1) or (B,)
        
        Returns:
            context: encoding, shape (B, output_dim)
        """
        # Embed discrete conditions
        drug_emb = self.drug_embedding(drug_ids)  # (B, embedding_dim)
        cell_line_emb = self.cell_line_embedding(cell_line_ids)  # (B, embedding_dim)
        
        # Encode dose (if provided)
        if doses is not None:
            if doses.dim() == 1:
                doses = doses.unsqueeze(-1)  # (B,) → (B, 1)
            dose_emb = self.dose_encoder(doses)  # (B, dose_embedding_dim)
            context = torch.cat([drug_emb, cell_line_emb, dose_emb], dim=-1)
        else:
            context = torch.cat([drug_emb, cell_line_emb], dim=-1)
        
        return context
    
    def get_condition_ids(self, drug: str, cell_line: str, drug_vocab: Dict, cell_line_vocab: Dict) -> tuple:
        """
        Convert string conditions to integer IDs.
        
        Args:
            drug: drug name
            cell_line: cell line name
            drug_vocab: dict mapping drug names to IDs
            cell_line_vocab: dict mapping cell line names to IDs
        
        Returns:
            (drug_id, cell_line_id)
        """
        drug_id = drug_vocab.get(drug, 0)
        cell_line_id = cell_line_vocab.get(cell_line, 0)
        
        return drug_id, cell_line_id


class ConditionVocabulary:
    """Manage vocabularies for categorical conditions."""
    
    def __init__(self):
        self.drug_vocab = {}
        self.cell_line_vocab = {}
        self.drug_idx = 0
        self.cell_line_idx = 0
    
    def add_drug(self, drug_name: str) -> int:
        """Add drug to vocabulary, return ID."""
        if drug_name not in self.drug_vocab:
            self.drug_vocab[drug_name] = self.drug_idx
            self.drug_idx += 1
        return self.drug_vocab[drug_name]
    
    def add_cell_line(self, cell_line_name: str) -> int:
        """Add cell line to vocabulary, return ID."""
        if cell_line_name not in self.cell_line_vocab:
            self.cell_line_vocab[cell_line_name] = self.cell_line_idx
            self.cell_line_idx += 1
        return self.cell_line_vocab[cell_line_name]
    
    def get_drug_id(self, drug_name: str) -> int:
        return self.drug_vocab.get(drug_name, 0)
    
    def get_cell_line_id(self, cell_line_name: str) -> int:
        return self.cell_line_vocab.get(cell_line_name, 0)
    
    def get_n_drugs(self) -> int:
        return len(self.drug_vocab)
    
    def get_n_cell_lines(self) -> int:
        return len(self.cell_line_vocab)


def encode_conditions_from_metadata(
    metadata: Dict[str, List],
    context_encoder: ContextEncoder,
    condition_vocab: ConditionVocabulary,
) -> torch.Tensor:
    """
    Encode condition metadata into context vectors.
    
    Args:
        metadata: dict with keys 'drugs', 'cell_lines', 'doses'
                 each value is list of length n_samples
        context_encoder: ContextEncoder instance
        condition_vocab: ConditionVocabulary instance
    
    Returns:
        context: shape (n_samples, context_dim)
    """
    drugs = metadata.get('drugs', [])
    cell_lines = metadata.get('cell_lines', [])
    doses = metadata.get('doses', None)
    
    n_samples = len(drugs)
    
    # Convert to IDs
    drug_ids = torch.tensor(
        [condition_vocab.get_drug_id(d) for d in drugs],
        dtype=torch.long
    )
    cell_line_ids = torch.tensor(
        [condition_vocab.get_cell_line_id(c) for c in cell_lines],
        dtype=torch.long
    )
    
    if doses is not None:
        doses_tensor = torch.tensor(doses, dtype=torch.float32)
    else:
        doses_tensor = None
    
    # Encode
    context = context_encoder(drug_ids, cell_line_ids, doses_tensor)
    
    return context


if __name__ == "__main__":
    # Test
    n_drugs = 3
    n_cell_lines = 2
    
    encoder = ContextEncoder(n_drugs, n_cell_lines, embedding_dim=8, dose_embedding_dim=4)
    
    B = 10
    drug_ids = torch.randint(0, n_drugs, (B,))
    cell_line_ids = torch.randint(0, n_cell_lines, (B,))
    doses = torch.rand(B, 1)
    
    context = encoder(drug_ids, cell_line_ids, doses)
    print(f"Context shape: {context.shape}")
    print(f"Expected: (10, {2*8+4})")
    
    # Test vocabulary
    vocab = ConditionVocabulary()
    vocab.add_drug("drug_A")
    vocab.add_drug("drug_B")
    vocab.add_cell_line("HCT116")
    vocab.add_cell_line("A549")
    
    print(f"Drug vocab: {vocab.drug_vocab}")
    print(f"Cell line vocab: {vocab.cell_line_vocab}")
