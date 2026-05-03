"""
Prepare train/test splits for multi-class video classification benchmark.

This script creates reproducible random splits from metadata.csv:
- 300 samples for training
- 100 samples for testing

The splits are stratified by subtype to ensure balanced representation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import os


def prepare_multiclass_splits(
    metadata_path: str,
    output_dir: str,
    train_size: int = 300,
    test_size: int = 100,
    random_seed: int = 42
):
    """
    Prepare stratified train/test splits for multi-class classification.
    
    Args:
        metadata_path: Path to metadata.csv
        output_dir: Directory to save split files
        train_size: Number of training samples
        test_size: Number of test samples
        random_seed: Random seed for reproducibility
    
    Multi-class labels:
    - diagnostic_class: non_rd, rd (2 classes)
    - subtype: normal, pvd, macula_intact, macula_detached (4 classes)
    """
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Load metadata
    print(f"Loading metadata from {metadata_path}")
    df = pd.read_csv(metadata_path)
    
    print(f"\nTotal samples: {len(df)}")
    print("\nClass distribution:")
    print(df.groupby(['diagnostic_class', 'subtype']).size())
    
    # Create label mappings
    diagnostic_classes = sorted(df['diagnostic_class'].unique())
    subtypes = sorted(df['subtype'].unique())
    
    diagnostic_to_idx = {cls: idx for idx, cls in enumerate(diagnostic_classes)}
    subtype_to_idx = {cls: idx for idx, cls in enumerate(subtypes)}
    
    print(f"\nDiagnostic classes: {diagnostic_classes}")
    print(f"Diagnostic mapping: {diagnostic_to_idx}")
    print(f"\nSubtypes: {subtypes}")
    print(f"Subtype mapping: {subtype_to_idx}")
    
    # Stratified split by subtype
    total_needed = train_size + test_size
    
    # First, sample total_needed samples stratified by subtype
    df_sampled, _ = train_test_split(
        df,
        train_size=total_needed,
        stratify=df['subtype'],
        random_state=random_seed
    )
    
    # Then split into train and test
    train_df, test_df = train_test_split(
        df_sampled,
        train_size=train_size,
        test_size=test_size,
        stratify=df_sampled['subtype'],
        random_state=random_seed
    )
    
    print(f"\nTrain set: {len(train_df)} samples")
    print(train_df.groupby(['diagnostic_class', 'subtype']).size())
    
    print(f"\nTest set: {len(test_df)} samples")
    print(test_df.groupby(['diagnostic_class', 'subtype']).size())
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save train split
    train_file = os.path.join(output_dir, 'train.txt')
    with open(train_file, 'w') as f:
        for _, row in train_df.iterrows():
            diagnostic_label = diagnostic_to_idx[row['diagnostic_class']]
            subtype_label = subtype_to_idx[row['subtype']]
            f.write(f"{row['file_path']},{diagnostic_label},{subtype_label}\n")
    
    print(f"\nSaved train split to {train_file}")
    
    # Save test split
    test_file = os.path.join(output_dir, 'test.txt')
    with open(test_file, 'w') as f:
        for _, row in test_df.iterrows():
            diagnostic_label = diagnostic_to_idx[row['diagnostic_class']]
            subtype_label = subtype_to_idx[row['subtype']]
            f.write(f"{row['file_path']},{diagnostic_label},{subtype_label}\n")
    
    print(f"Saved test split to {test_file}")
    
    # Save label mappings
    mappings = {
        'diagnostic_classes': diagnostic_classes,
        'diagnostic_to_idx': diagnostic_to_idx,
        'subtypes': subtypes,
        'subtype_to_idx': subtype_to_idx,
        'num_diagnostic_classes': len(diagnostic_classes),
        'num_subtype_classes': len(subtypes),
        'train_size': len(train_df),
        'test_size': len(test_df),
        'random_seed': random_seed
    }
    
    mappings_file = os.path.join(output_dir, 'label_mappings.json')
    with open(mappings_file, 'w') as f:
        json.dump(mappings, f, indent=2)
    
    print(f"Saved label mappings to {mappings_file}")
    
    # Save detailed statistics
    # Convert tuple keys to strings for JSON serialization
    train_dist = train_df.groupby(['diagnostic_class', 'subtype']).size()
    test_dist = test_df.groupby(['diagnostic_class', 'subtype']).size()
    
    stats = {
        'train_distribution': {f"{k[0]}_{k[1]}": int(v) for k, v in train_dist.items()},
        'test_distribution': {f"{k[0]}_{k[1]}": int(v) for k, v in test_dist.items()}
    }
    
    stats_file = os.path.join(output_dir, 'split_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Saved statistics to {stats_file}")
    
    return train_df, test_df, mappings


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare multi-class splits')
    parser.add_argument('--metadata', type=str, 
                       default='../../erdes/metadata.csv',
                       help='Path to metadata.csv')
    parser.add_argument('--output_dir', type=str,
                       default='./splits',
                       help='Output directory for splits')
    parser.add_argument('--train_size', type=int, default=300,
                       help='Number of training samples')
    parser.add_argument('--test_size', type=int, default=100,
                       help='Number of test samples')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    prepare_multiclass_splits(
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        train_size=args.train_size,
        test_size=args.test_size,
        random_seed=args.seed
    )
