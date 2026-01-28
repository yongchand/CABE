import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
import torch
import re


class DrugDiscoveryDatasetEmb(Dataset):
    """
    Dataset for drug discovery with expert scores and molecular embeddings.
    Uses explicit train/valid/test splits via PDB ID lists.
    """
    def __init__(self, csv_path, split='train', seed=42, normalization_stats=None,
                 train_pdb_ids=None, valid_pdb_ids=None, test_pdb_ids=None):
        super(DrugDiscoveryDatasetEmb, self).__init__()
        
        # Load data
        df = pd.read_csv(csv_path)
        
        if split == 'train':
            print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # Extract embedding columns (Emb_0 to Emb_703)
        emb_cols = [col for col in df.columns if col.startswith('Emb_')]
        if split == 'train':
            print(f"Found {len(emb_cols)} embedding dimensions")
        
        # Expert score columns
        self.expert_cols = ['GNINA_Affinity', 'BIND_pIC50', 'flowdock_score', 'DynamicBind_score']
        
        # Parse binding affinity labels and get valid indices
        labels, affinity_valid_mask = self._parse_affinity(df['Binding_Affinity'].values)
        
        # Filter out invalid samples (both affinity parsing and NaN expert scores)
        expert_nan_mask = ~df[self.expert_cols].isna().any(axis=1).values
        valid_mask = affinity_valid_mask & expert_nan_mask
        
        valid_indices = np.where(valid_mask)[0]
        if split == 'train':
            print(f"Valid samples: {len(valid_indices)} / {len(df)} ({len(valid_indices)/len(df)*100:.1f}%)")
        
        # Extract features for valid samples only
        embeddings = df[emb_cols].values[valid_indices].astype(np.float32)
        expert_scores = df[self.expert_cols].values[valid_indices].astype(np.float32)
        labels = labels[valid_mask]
        complex_ids = df['ComplexID'].values[valid_indices]
        
        # Get the PDB IDs for the requested split
        if split == 'train':
            pdb_ids = train_pdb_ids
        elif split == 'valid':
            pdb_ids = valid_pdb_ids
        elif split == 'test':
            pdb_ids = test_pdb_ids
        else:
            raise ValueError(f"Unknown split: {split}. Must be 'train', 'valid', or 'test'")
        
        if pdb_ids is None:
            raise ValueError(f"PDB IDs must be provided for {split} split")
        
        # Filter to only include PDBs that exist in this split
        pdb_set = set(str(pdb_id).lower() for pdb_id in pdb_ids)
        split_mask = np.array([str(cid).lower() in pdb_set for cid in complex_ids])
        
        if split == 'train':
            print(f"Split '{split}': requested {len(pdb_ids)} PDBs, found {np.sum(split_mask)} in data")
        
        # Apply filter
        embeddings = embeddings[split_mask]
        expert_scores = expert_scores[split_mask]
        labels = labels[split_mask]
        complex_ids = complex_ids[split_mask]
        
        if len(embeddings) == 0:
            raise ValueError(f"No samples found for {split} split")
        
        # Normalization
        if normalization_stats is not None:
            self.emb_mean = np.asarray(normalization_stats['mean'], dtype=np.float32)
            self.emb_std = np.asarray(normalization_stats['std'], dtype=np.float32)
        else:
            if split != 'train':
                raise ValueError("Normalization stats must be provided for non-training splits")
            self.emb_mean = embeddings.mean(axis=0).astype(np.float32)
            self.emb_std = (embeddings.std(axis=0) + 1e-8).astype(np.float32)
        
        embeddings = (embeddings - self.emb_mean) / self.emb_std
        
        # Store tensors
        self.embeddings = torch.tensor(embeddings).cpu()
        self.expert_scores = torch.tensor(expert_scores).cpu()
        self.labels = torch.tensor(labels).cpu()
        self.complex_ids = complex_ids
        
        print(f"{split} set: {len(self.labels)} samples")
    
    def _parse_affinity(self, affinity_strings):
        """
        Parse binding affinity from strings like 'Kd=6.67uM', 'Ki=19uM'
        Convert to pKd/pKi values (negative log of molar concentration)
        """
        labels = []
        valid_mask = []
        
        for s in affinity_strings:
            try:
                s_str = str(s)
                
                # Skip inequality values
                if '>' in s_str or '<' in s_str:
                    labels.append(0.0)
                    valid_mask.append(False)
                    continue
                
                # Extract numeric value and unit
                match = re.search(r'([0-9.]+)([a-zA-Z]+)', s_str)
                if match:
                    value = float(match.group(1))
                    unit = match.group(2).lower()
                    
                    # Convert to Molar
                    if 'nm' in unit:
                        molar = value * 1e-9
                    elif 'um' in unit or 'μm' in unit:
                        molar = value * 1e-6
                    elif 'mm' in unit:
                        molar = value * 1e-3
                    elif 'pm' in unit:
                        molar = value * 1e-12
                    elif 'm' in unit:
                        molar = value
                    else:
                        labels.append(0.0)
                        valid_mask.append(False)
                        continue
                    
                    p_value = -np.log10(molar)
                    labels.append(p_value)
                    valid_mask.append(True)
                else:
                    labels.append(0.0)
                    valid_mask.append(False)
            except Exception:
                labels.append(0.0)
                valid_mask.append(False)
        
        return np.array(labels, dtype=np.float32), np.array(valid_mask, dtype=bool)
    
    def get_dim(self):
        """Return dimensions of embeddings and expert scores"""
        return self.embeddings.shape[1], self.expert_scores.shape[1]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return (self.expert_scores[index], 
                self.embeddings[index]), self.labels[index], self.complex_ids[index]
