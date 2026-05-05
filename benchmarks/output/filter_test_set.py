#!/usr/bin/env python3
"""
Filter gemini_prediction_old.csv to only include rows that are in the test set.
"""
import pandas as pd
import sys

def main():
    # File paths (relative to script location)
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    gemini_predictions = os.path.join(script_dir, 'gpt_prediction_all.csv')
    test_csv = os.path.join(script_dir, '../input/balanced_split_desc_test.csv')
    output_csv = os.path.join(script_dir, 'gpt_prediction.csv')
    
    print('='*70)
    print('FILTERING GEMINI PREDICTIONS TO TEST SET')
    print('='*70)
    
    # Load the files
    print(f'\nLoading gemini predictions: {gemini_predictions}')
    df_gemini = pd.read_csv(gemini_predictions)
    print(f'  Total predictions: {len(df_gemini):,}')
    
    print(f'\nLoading test set: {test_csv}')
    df_test = pd.read_csv(test_csv)
    print(f'  Total test samples: {len(df_test):,}')
    
    # Get test set clip_ids
    test_clip_ids = set(df_test['clip_id'].values)
    print(f'\nUnique test clip_ids: {len(test_clip_ids):,}')
    
    # Filter gemini predictions to only include test set
    print('\nFiltering predictions...')
    df_filtered = df_gemini[df_gemini['clip_id'].isin(test_clip_ids)].copy()
    
    print(f'  Filtered predictions: {len(df_filtered):,}')
    print(f'  Removed: {len(df_gemini) - len(df_filtered):,}')
    
    # Save filtered results
    print(f'\nSaving to: {output_csv}')
    df_filtered.to_csv(output_csv, index=False)
    
    print('\n' + '='*70)
    print('FILTERING COMPLETE!')
    print('='*70)
    print(f'\nOutput file: {output_csv}')
    print(f'Rows in output: {len(df_filtered):,}')
    
    # Show some statistics
    print('\n' + '='*70)
    print('FILTERED DATA STATISTICS')
    print('='*70)
    
    if 'diagnostic_class' in df_filtered.columns:
        print('\nDiagnostic class distribution:')
        print(df_filtered['diagnostic_class'].value_counts())
    
    if 'subtype' in df_filtered.columns:
        print('\nSubtype distribution:')
        print(df_filtered['subtype'].value_counts())
    
    # Check for any missing clip_ids
    gemini_clip_ids = set(df_gemini['clip_id'].values)
    missing_in_gemini = test_clip_ids - gemini_clip_ids
    
    if missing_in_gemini:
        print(f'\n⚠️  WARNING: {len(missing_in_gemini)} test set clip_ids not found in gemini predictions:')
        for clip_id in sorted(list(missing_in_gemini))[:10]:
            print(f'  - {clip_id}')
        if len(missing_in_gemini) > 10:
            print(f'  ... and {len(missing_in_gemini) - 10} more')
    else:
        print('\n✓ All test set clip_ids found in gemini predictions')

if __name__ == '__main__':
    main()
