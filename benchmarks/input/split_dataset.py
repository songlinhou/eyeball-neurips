#!/usr/bin/env python3
"""
Split balanced_split_desc.csv into train and test sets.
Uses stratified split to maintain class balance.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Configuration
INPUT_CSV = "balanced_split_desc.csv"
TRAIN_CSV = "balanced_split_desc_train.csv"
TEST_CSV = "balanced_split_desc_test.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def main():
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, INPUT_CSV)
    train_path = os.path.join(script_dir, TRAIN_CSV)
    test_path = os.path.join(script_dir, TEST_CSV)
    
    print(f"Loading CSV from: {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"Total samples: {len(df)}")
    print(f"\nClass distribution:")
    print(df['diagnostic_class'].value_counts())
    print(f"\nSubtype distribution:")
    print(df['subtype'].value_counts())
    
    # Stratified split by diagnostic_class and subtype
    # Create combined stratification column
    df['stratify_col'] = df['diagnostic_class'].astype(str) + '_' + df['subtype'].astype(str)
    
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['stratify_col']
    )
    
    # Remove temporary stratification column
    train_df = train_df.drop('stratify_col', axis=1)
    test_df = test_df.drop('stratify_col', axis=1)
    
    print(f"\n{'='*60}")
    print(f"Split complete!")
    print(f"{'='*60}")
    print(f"Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Test set:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    print(f"\nTrain set class distribution:")
    print(train_df['diagnostic_class'].value_counts())
    print(f"\nTest set class distribution:")
    print(test_df['diagnostic_class'].value_counts())
    
    # Save splits
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Files saved:")
    print(f"  Train: {train_path}")
    print(f"  Test:  {test_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
