import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
import torch
import re


class DrugDiscoveryDatasetEmb(Dataset):
    """
    Dataset for drug discovery with expert scores and molecular embeddings.

    Supports:
    - Using test IDs from data/test.csv as test set
    - Random split for train/valid from remaining data
    """
    def __init__(
        self,
        csv_path,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        seed: int = 42,
        normalization_stats=None,
        test_ids_path: str = "data/test.csv",
    ):
        super().__init__()

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

        # Load test IDs from test.csv
        try:
            test_df = pd.read_csv(test_ids_path)
            test_ids_set = set(test_df['name'].str.lower().str.strip())
            if split == "train":
                print(f"Loaded {len(test_ids_set)} test IDs from {test_ids_path}")
        except Exception as e:
            if split == "train":
                print(f"Warning: Could not load test IDs from {test_ids_path}: {e}")
                print("Falling back to random split")
            test_ids_set = set()

        # Split data based on test IDs
        if len(test_ids_set) > 0:
            # Create mask for test set
            test_mask = np.array([str(cid).lower().strip() in test_ids_set for cid in complex_ids])
            non_test_mask = ~test_mask
            
            if split == "train":
                print(f"Test set IDs found: {np.sum(test_mask)} samples")
                print(f"Remaining for train/valid: {np.sum(non_test_mask)} samples")
            
            # Split train/valid from non-test data
            if split in ["train", "valid"]:
                # Use only non-test data for train/valid split
                non_test_indices = np.where(non_test_mask)[0]
                np.random.seed(seed)
                n_non_test = len(non_test_indices)
                indices = np.random.permutation(n_non_test)
                
                train_end = int(train_ratio * n_non_test)
                if train_end == 0:
                    train_end = max(1, n_non_test)
                val_end = train_end + int(val_ratio * n_non_test)
                val_end = min(val_end, n_non_test)
                
                train_indices_local = indices[:train_end]
                val_indices_local = indices[train_end:val_end]
                
                # Map back to original indices
                train_indices = non_test_indices[train_indices_local]
                val_indices = non_test_indices[val_indices_local]
                test_indices = np.where(test_mask)[0]
                
                if split == "train":
                    print(
                        f"Split: train={len(train_indices)}, "
                        f"valid={len(val_indices)}, test={len(test_indices)}"
                    )
            elif split == "test":
                # Use test set
                test_indices = np.where(test_mask)[0]
                train_indices = np.array([], dtype=np.int64)
                val_indices = np.array([], dtype=np.int64)
                
                if len(test_indices) == 0:
                    raise ValueError(f"No test samples found matching IDs in {test_ids_path}")
        else:
            # Fallback to random split if test IDs not available
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
                    f"Random split (fallback): train={len(train_indices)}, "
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
            idx = train_indices
        elif split == "valid":
            idx = val_indices
        elif split == "test":
            idx = test_indices
        else:
            raise ValueError("split must be one of 'train', 'valid', 'test'")

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
        """
        labels = []
        valid_mask = []

        for s in affinity_strings:
            try:
                match = re.search(r"([0-9.]+)([a-zA-Z]+)", str(s))
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