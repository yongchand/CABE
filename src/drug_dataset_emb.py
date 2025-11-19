import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
import torch
import re
import json
import os


class DrugDiscoveryDatasetEmb(Dataset):
    """
    Dataset for drug discovery with expert scores and molecular embeddings.

    Supports:
    - Using predefined splits from split.json
    - Train/validation: 80/20 random split on "train" set from split.json
    - Test: casf2016_indep + casf2013_indep from split.json
    - CASF2016: casf2016 set from split.json
    - CASF2013: casf2013 set from split.json
    """
    def __init__(
        self,
        csv_path,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        seed: int = 42,
        normalization_stats=None,
        casf2016_pdb_ids=None,
        split_json_path: str = "split.json",
    ):
        super().__init__()

        # Load split.json if it exists
        split_data = None
        if os.path.isfile(split_json_path):
            with open(split_json_path, "r", encoding="utf-8") as f:
                split_data = json.load(f)
            if split == "train":
                print(f"Loaded splits from {split_json_path}")
        elif split == "train":
            print(f"Warning: {split_json_path} not found, falling back to random splits")

        # Load data
        df = pd.read_csv(csv_path)

        # Only print CSV loading info for train split to avoid duplicate output
        if split == "train":
            print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")

        # Embedding columns (Emb_0 ... Emb_703)
        emb_cols = [c for c in df.columns if c.startswith("Emb_")]
        if split == "train":
            print(f"Found {len(emb_cols)} embedding dimensions")

        # Expert score columns (raw, un-normalized)
        self.expert_cols = ["GNINA_Affinity", "BIND_pIC50", "flowdock_score"]

        # Parse binding affinity into pKd / pKi
        labels, valid_mask = self._parse_affinity(df["Binding_Affinity"].values)

        # Filter to valid rows
        valid_indices = np.where(valid_mask)[0]
        if split == "train":
            print(
                f"Valid samples after parsing affinity: "
                f"{len(valid_indices)} / {len(df)} "
                f"({len(valid_indices) / len(df) * 100:.1f}%)"
            )

        embeddings = df.loc[valid_indices, emb_cols].values.astype(np.float32)
        expert_scores = df.loc[valid_indices, self.expert_cols].values.astype(np.float32)
        labels = labels[valid_mask]
        complex_ids = df["ComplexID"].values[valid_indices]
        year_folders = (
            df["YearFolder"].values[valid_indices]
            if "YearFolder" in df.columns
            else None
        )

        # Initialize split indices
        train_indices = np.array([], dtype=np.int64)
        val_indices = np.array([], dtype=np.int64)
        test_indices = np.array([], dtype=np.int64)

        # Use split.json if available
        if split_data is not None:
            # Create sets for each split from split.json
            casf2016_set = {str(pid).lower() for pid in split_data.get("casf2016", [])}
            casf2013_set = {str(pid).lower() for pid in split_data.get("casf2013", [])}
            casf2016_indep_set = {str(pid).lower() for pid in split_data.get("casf2016_indep", [])}
            casf2013_indep_set = {str(pid).lower() for pid in split_data.get("casf2013_indep", [])}
            train_set = {str(pid).lower() for pid in split_data.get("train", [])}
            
            # Test set is combination of casf2016_indep and casf2013_indep
            test_set = casf2016_indep_set | casf2013_indep_set
            
            # Create masks for each split
            casf2016_indep_mask = np.array([str(cid).lower() in casf2016_indep_set for cid in complex_ids])
            casf2013_indep_mask = np.array([str(cid).lower() in casf2013_indep_set for cid in complex_ids])
            test_mask = np.array([str(cid).lower() in test_set for cid in complex_ids])
            train_set_mask = np.array([str(cid).lower() in train_set for cid in complex_ids])
            
            if split == "casf2016":
                # Use casf2016_indep set (independent test set)
                mask = casf2016_indep_mask
                print(f"Including {np.sum(mask)} CASF 2016 independent samples from split.json")
                if np.sum(mask) == 0:
                    raise ValueError("No CASF 2016 independent samples found in dataset")
                
                embeddings = embeddings[mask]
                expert_scores = expert_scores[mask]
                labels = labels[mask]
                complex_ids = complex_ids[mask]
                if year_folders is not None:
                    year_folders = year_folders[mask]
                valid_indices = valid_indices[mask]
                
            elif split == "casf2013":
                # Use casf2013_indep set (independent test set)
                mask = casf2013_indep_mask
                print(f"Including {np.sum(mask)} CASF 2013 independent samples from split.json")
                if np.sum(mask) == 0:
                    raise ValueError("No CASF 2013 independent samples found in dataset")
                
                embeddings = embeddings[mask]
                expert_scores = expert_scores[mask]
                labels = labels[mask]
                complex_ids = complex_ids[mask]
                if year_folders is not None:
                    year_folders = year_folders[mask]
                valid_indices = valid_indices[mask]
                
            elif split == "test":
                # Use test set (casf2016_indep + casf2013_indep)
                mask = test_mask
                print(f"Including {np.sum(mask)} test samples (casf2016_indep + casf2013_indep) from split.json")
                if np.sum(mask) == 0:
                    raise ValueError("No test samples found in dataset")
                
                embeddings = embeddings[mask]
                expert_scores = expert_scores[mask]
                labels = labels[mask]
                complex_ids = complex_ids[mask]
                if year_folders is not None:
                    year_folders = year_folders[mask]
                valid_indices = valid_indices[mask]
                
            else:
                # For train/valid, use train set and exclude only test sets (casf2016_indep + casf2013_indep)
                # CASF 2016 and CASF 2013 are separate evaluation sets but can be used for training
                exclude_mask = test_mask
                train_set_only_mask = train_set_mask & ~exclude_mask
                
                if split == "train":
                    print(f"Using train set from split.json: {np.sum(train_set_only_mask)} samples")
                    print(f"Excluding {np.sum(test_mask)} test samples (casf2016_indep + casf2013_indep)")
                
                embeddings = embeddings[train_set_only_mask]
                expert_scores = expert_scores[train_set_only_mask]
                labels = labels[train_set_only_mask]
                complex_ids = complex_ids[train_set_only_mask]
                if year_folders is not None:
                    year_folders = year_folders[train_set_only_mask]
                valid_indices = valid_indices[train_set_only_mask]
                
                # Random 90/10 split on train set for train/validation
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
                
                if split == "train":
                    print(
                        f"Random 80/20 split on train set: train={len(train_indices)}, "
                        f"valid={len(val_indices)}"
                    )
        else:
            # Fallback to old logic if split.json not found
            # Handle CASF2016 logic
            if split == "casf2016":
                if casf2016_pdb_ids is None:
                    raise ValueError("CASF 2016 PDB IDs must be provided for casf2016 split")
                casf_set = {str(pid).lower() for pid in casf2016_pdb_ids}
                casf_mask = np.array([str(cid).lower() in casf_set for cid in complex_ids])
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

            elif casf2016_pdb_ids is not None:
                casf_set = {str(pid).lower() for pid in casf2016_pdb_ids}
                non_casf_mask = np.array(
                    [str(cid).lower() not in casf_set for cid in complex_ids]
                )
                if split == "train":
                    print(f"Excluding {np.sum(~non_casf_mask)} CASF 2016 samples")

                embeddings = embeddings[non_casf_mask]
                expert_scores = expert_scores[non_casf_mask]
                labels = labels[non_casf_mask]
                complex_ids = complex_ids[non_casf_mask]
                if year_folders is not None:
                    year_folders = year_folders[non_casf_mask]
                valid_indices = valid_indices[non_casf_mask]

            # Random split for non-CASF2016 splits
            if split != "casf2016":
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

                if split == "train":
                    print(
                        f"Random split: train={len(train_indices)}, "
                        f"valid={len(val_indices)}, test={len(test_indices)}"
                    )

        # --- Embedding normalization (z-score) ---
        # Only embeddings are normalized here. Expert scores are left raw and
        # normalized inside the model using dataset-level statistics.
        if normalization_stats is not None:
            self.emb_mean = np.asarray(normalization_stats["mean"], dtype=np.float32)
            self.emb_std = np.asarray(normalization_stats["std"], dtype=np.float32)
        else:
            if split != "train":
                raise ValueError(
                    "Normalization stats must be provided for non-training splits"
                )
            train_emb = embeddings[train_indices]
            self.emb_mean = train_emb.mean(axis=0).astype(np.float32)
            self.emb_std = (train_emb.std(axis=0) + 1e-8).astype(np.float32)

        embeddings = (embeddings - self.emb_mean) / self.emb_std

        # Select indices per split
        if split == "train":
            if len(train_indices) > 0:
                idx = train_indices
            else:
                # Fallback: use all indices (for backward compatibility or when split.json not used)
                idx = np.arange(len(embeddings), dtype=np.int64)
        elif split == "valid":
            if len(val_indices) > 0:
                idx = val_indices
            else:
                raise ValueError("Validation indices not available. Make sure split.json is used.")
        elif split == "test":
            if len(test_indices) > 0:
                idx = test_indices
            else:
                # If using split.json, all indices are already filtered to test set
                idx = np.arange(len(embeddings), dtype=np.int64)
        elif split == "casf2016" or split == "casf2013":
            idx = np.arange(len(embeddings), dtype=np.int64)
        else:
            raise ValueError("split must be one of 'train', 'valid', 'test', 'casf2016', 'casf2013'")

        # Final tensors (keep on CPU; training script moves to device)
        self.embeddings = torch.tensor(embeddings[idx]).cpu()
        self.expert_scores = torch.tensor(expert_scores[idx]).cpu()
        self.labels = torch.tensor(labels[idx]).cpu()
        self.complex_ids = complex_ids[idx]

        print(f"{split} set: {len(self.labels)} samples")
        print(f"  Embeddings: {self.embeddings.shape}")
        print(f"  Expert scores: {self.expert_scores.shape}")
        if len(self.labels) > 0:
            print(
                f"  Label range: [{self.labels.min():.2f}, {self.labels.max():.2f}]"
            )
        else:
            print("  Label range: [N/A - empty dataset]")

    def _parse_affinity(self, affinity_strings):
        """
        Parse binding affinity from strings like 'Kd=6.67uM', 'Ki=19uM'
        Convert to pKd/pKi values (negative log of molar concentration)
        
        Drops ambiguous labels with inequality operators (>, <, >=, <=) and non-numeric values.
        """
        labels = []
        valid_mask = []

        for s in affinity_strings:
            try:
                s_str = str(s).strip()
                
                # Check for inequality operators (>, <, >=, <=) - these are ambiguous and should be dropped
                # We check for > or < followed optionally by =, but not just = alone (which is valid in "Kd=6.67uM")
                if re.search(r'[><][=]?', s_str) or re.search(r'[><]\s', s_str):
                    labels.append(0.0)
                    valid_mask.append(False)
                    continue
                
                match = re.search(r"([0-9.]+)([a-zA-Z]+)", s_str)
                if not match:
                    labels.append(0.0)
                    valid_mask.append(False)
                    continue

                value = float(match.group(1))
                unit = match.group(2).lower()

                if "nm" in unit:
                    molar = value * 1e-9
                elif "um" in unit or "μm" in unit:
                    molar = value * 1e-6
                elif "mm" in unit:
                    molar = value * 1e-3
                elif "pm" in unit:
                    molar = value * 1e-12
                elif unit == "m":
                    molar = value
                else:
                    labels.append(0.0)
                    valid_mask.append(False)
                    continue

                p_val = -np.log10(molar)
                labels.append(p_val)
                valid_mask.append(True)
            except Exception:
                labels.append(0.0)
                valid_mask.append(False)

        labels = np.asarray(labels, dtype=np.float32)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        return labels, valid_mask

    def get_dim(self):
        """Return (embedding_dim, num_experts)."""
        return self.embeddings.shape[1], self.expert_scores.shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.expert_scores[idx],
            self.embeddings[idx],
        ), self.labels[idx], self.complex_ids[idx]