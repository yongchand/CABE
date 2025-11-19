import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
import torch
import re


class DrugDiscoveryDatasetEmb(Dataset):
    """
    Dataset for drug discovery with expert scores and molecular embeddings.
    
    Supports:
    - Excluding CASF 2016 dataset for separate test set
    - Random 80/10/10 split ratio (train/valid/test)
    """
    def __init__(self, csv_path, split='train', train_ratio=0.8, val_ratio=0.1,
                 seed=42, normalization_stats=None, casf2016_pdb_ids=None):
        super(DrugDiscoveryDatasetEmb, self).__init__()
        
        # Load data
        df = pd.read_csv(csv_path)
        
        # Only print CSV loading info for train split to avoid duplicate output
        if split == 'train':
            print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # Extract embedding columns (Emb_0 to Emb_703)
        emb_cols = [col for col in df.columns if col.startswith('Emb_')]
        if split == 'train':
            print(f"Found {len(emb_cols)} embedding dimensions")
        
        # Expert score columns
        self.expert_cols = ['GNINA_Affinity', 'BIND_pIC50', 'flowdock_score']
        
        # Parse binding affinity labels and get valid indices
        labels, valid_mask = self._parse_affinity(df['Binding_Affinity'].values)
        
        # Filter out invalid samples
        valid_indices = np.where(valid_mask)[0]
        if split == 'train':
            print(f"Valid samples after parsing affinity: {len(valid_indices)} / {len(df)} "
                  f"({len(valid_indices)/len(df)*100:.1f}%)")
        
        # Extract features for valid samples only
        embeddings = df[emb_cols].values[valid_indices].astype(np.float32)
        expert_scores = df[self.expert_cols].values[valid_indices].astype(np.float32)
        labels = labels[valid_mask]
        complex_ids = df['ComplexID'].values[valid_indices]
        year_folders = df['YearFolder'].values[valid_indices] if 'YearFolder' in df.columns else None
        
        # Handle CASF 2016: exclude for train/val/test, include only for casf2016 split
        if split == 'casf2016':
            # For CASF 2016 split, we want ONLY CASF 2016 complexes
            if casf2016_pdb_ids is None:
                raise ValueError("CASF 2016 PDB IDs must be provided for casf2016 split")
            casf2016_set = set(str(pdb_id).lower() for pdb_id in casf2016_pdb_ids)
            casf_mask = np.array([str(cid).lower() in casf2016_set for cid in complex_ids])
            print(f"Including {np.sum(casf_mask)} CASF 2016 samples")
            
            if np.sum(casf_mask) == 0:
                raise ValueError("No CASF 2016 samples found in dataset")
            
            embeddings = embeddings[casf_mask]
            expert_scores = expert_scores[casf_mask]
            labels = labels[casf_mask]
            complex_ids = complex_ids[casf_mask]
            if year_folders is not None:
                year_folders = year_folders[casf_mask]
            valid_indices = valid_indices[casf_mask]
            
            # For CASF 2016, we don't need to split - use all samples
            train_indices = np.array([], dtype=np.int64)
            val_indices = np.array([], dtype=np.int64)
            test_indices = np.array([], dtype=np.int64)
        elif casf2016_pdb_ids is not None:
            # Exclude CASF 2016 for train/val/test splits
            casf2016_set = set(str(pdb_id).lower() for pdb_id in casf2016_pdb_ids)
            non_casf_mask = np.array([str(cid).lower() not in casf2016_set for cid in complex_ids])
            if split == 'train':
                print(f"Excluding {np.sum(~non_casf_mask)} CASF 2016 samples")
            
            embeddings = embeddings[non_casf_mask]
            expert_scores = expert_scores[non_casf_mask]
            labels = labels[non_casf_mask]
            complex_ids = complex_ids[non_casf_mask]
            if year_folders is not None:
                year_folders = year_folders[non_casf_mask]
            valid_indices = valid_indices[non_casf_mask]
        
        # Random split (80/10/10) - skip for CASF 2016 split
        if split != 'casf2016':
            np.random.seed(seed)
            n_samples = len(valid_indices)
            indices = np.random.permutation(n_samples)
            
            train_end = int(train_ratio * n_samples)
            if train_end == 0:
                train_end = max(1, n_samples)
            val_end = train_end + int(val_ratio * n_samples)
            val_end = min(val_end, n_samples)
            
            train_indices = indices[:train_end]
            val_indices = indices[train_end:val_end]
            test_indices = indices[val_end:]
            
            # Only print split info for train split to avoid duplicate output
            if split == 'train':
                print(f"Random split: train={len(train_indices)}, valid={len(val_indices)}, test={len(test_indices)}")
        
        # Fit or reuse normalization stats
        # IMPORTANT: normalization_stats must always come from TRAINING data only
        # to prevent data leakage. Stats should be computed from train split and
        # reused for valid/test/casf2016 splits.
        if normalization_stats is not None:
            self.emb_mean = np.asarray(normalization_stats['mean'], dtype=np.float32)
            self.emb_std = np.asarray(normalization_stats['std'], dtype=np.float32)
        else:
            if split not in ['train', 'casf2016']:
                raise ValueError("Normalization stats must be provided for non-training splits")
            if split == 'casf2016':
                raise ValueError("Normalization stats must be provided for casf2016 split (use stats from training)")
            # Compute normalization stats from TRAINING data only
            train_embeddings = embeddings[train_indices]
            self.emb_mean = train_embeddings.mean(axis=0).astype(np.float32)
            self.emb_std = (train_embeddings.std(axis=0) + 1e-8).astype(np.float32)
        
        # Normalize ALL embeddings using training stats (before split selection)
        # This ensures consistent normalization across all splits
        embeddings = (embeddings - self.emb_mean) / self.emb_std
        
        # Select indices for current split
        if split == 'train':
            idx = train_indices
        elif split == 'valid':
            idx = val_indices
        elif split == 'test':
            idx = test_indices
        elif split == 'casf2016':
            # For CASF 2016, use all samples (already filtered above)
            idx = np.arange(len(embeddings), dtype=np.int64)
        else:
            raise ValueError(f"Unknown split: {split}. Must be 'train', 'valid', 'test', or 'casf2016'")
        
        # Store tensors
        self.embeddings = torch.tensor(embeddings[idx]).cpu()
        self.expert_scores = torch.tensor(expert_scores[idx]).cpu()
        self.labels = torch.tensor(labels[idx]).cpu()
        self.complex_ids = complex_ids[idx]
        
        print(f"{split} set: {len(self.labels)} samples")
        print(f"  Embeddings: {self.embeddings.shape}")
        print(f"  Expert scores: {self.expert_scores.shape}")
        if len(self.labels) > 0:
            print(f"  Label range: [{self.labels.min():.2f}, {self.labels.max():.2f}]")
        else:
            print("  Label range: [N/A - empty dataset]")
    
    def _parse_affinity(self, affinity_strings):
        """
        Parse binding affinity from strings like 'Kd=6.67uM', 'Ki=19uM'
        Convert to pKd/pKi values (negative log of molar concentration)
        
        Returns:
            labels: numpy array of parsed values (only valid ones)
            valid_mask: boolean mask indicating which samples are valid
        """
        labels = []
        valid_mask = []
        
        for s in affinity_strings:
            try:
                # Extract numeric value and unit
                match = re.search(r'([0-9.]+)([a-zA-Z]+)', str(s))
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
                        # Skip if unit is unrecognized
                        labels.append(0.0)  # Placeholder
                        valid_mask.append(False)
                        continue
                    
                    # Convert to pKd/pKi
                    p_value = -np.log10(molar)
                    labels.append(p_value)
                    valid_mask.append(True)
                else:
                    # If can't parse, skip this sample
                    labels.append(0.0)  # Placeholder
                    valid_mask.append(False)
            except Exception:
                # If parsing fails, skip this sample
                labels.append(0.0)  # Placeholder
                valid_mask.append(False)
        
        labels = np.array(labels, dtype=np.float32)
        valid_mask = np.array(valid_mask, dtype=bool)
        
        return labels, valid_mask
    
    def get_dim(self):
        """Return dimensions of embeddings and expert scores"""
        return self.embeddings.shape[1], self.expert_scores.shape[1]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return (self.expert_scores[index], 
                self.embeddings[index]), self.labels[index], self.complex_ids[index]