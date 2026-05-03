import torch
import pandas as pd
from improved_training import train_with_gradual_unfreezing
import time


def compare_all_models():
    """
    Compare different model architectures and training strategies
    """
    results = []
    
    # Configuration matrix
    configs = [
        # (model_class, use_focal_loss, use_mixup, use_tta, description)
        ('improved', True, True, True, 'ImprovedResNet3D + Focal + Mixup + TTA'),
        ('improved', True, True, False, 'ImprovedResNet3D + Focal + Mixup'),
        ('improved', True, False, False, 'ImprovedResNet3D + Focal'),
        ('improved', False, False, False, 'ImprovedResNet3D (baseline)'),
        ('multiscale', True, True, True, 'MultiScaleResNet3D + Focal + Mixup + TTA'),
        ('auxiliary', True, True, True, 'ResNet3DWithAuxiliary + Focal + Mixup + TTA'),
    ]
    
    print("="*80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    print(f"\nTotal configurations to test: {len(configs)}\n")
    
    for idx, (model_class, use_focal, use_mixup, use_tta, description) in enumerate(configs):
        print(f"\n{'='*80}")
        print(f"Configuration {idx+1}/{len(configs)}: {description}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            model, val_acc, test_acc = train_with_gradual_unfreezing(
                model_class=model_class,
                num_epochs=30,
                use_focal_loss=use_focal,
                use_mixup=use_mixup,
                use_tta=use_tta,
                save_path=f'best_{model_class}_{idx}.pth'
            )
            
            training_time = time.time() - start_time
            
            results.append({
                'Configuration': description,
                'Model': model_class,
                'Focal Loss': use_focal,
                'Mixup': use_mixup,
                'TTA': use_tta,
                'Val Accuracy': val_acc,
                'Test Accuracy': test_acc,
                'Training Time (min)': training_time / 60
            })
            
            print(f"\n✓ Completed: Val={val_acc:.2f}%, Test={test_acc:.2f}%, Time={training_time/60:.1f}min")
            
        except Exception as e:
            print(f"\n✗ Failed: {str(e)}")
            results.append({
                'Configuration': description,
                'Model': model_class,
                'Focal Loss': use_focal,
                'Mixup': use_mixup,
                'TTA': use_tta,
                'Val Accuracy': 0,
                'Test Accuracy': 0,
                'Training Time (min)': 0
            })
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    
    # Sort by test accuracy
    df_results = df_results.sort_values('Test Accuracy', ascending=False)
    
    # Save results
    df_results.to_csv('model_comparison_results.csv', index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(df_results.to_string(index=False))
    
    print("\n" + "="*80)
    print("TOP 3 CONFIGURATIONS")
    print("="*80)
    for idx, row in df_results.head(3).iterrows():
        print(f"\n{idx+1}. {row['Configuration']}")
        print(f"   Val Accuracy: {row['Val Accuracy']:.2f}%")
        print(f"   Test Accuracy: {row['Test Accuracy']:.2f}%")
        print(f"   Training Time: {row['Training Time (min)']:.1f} minutes")
    
    return df_results


def quick_comparison():
    """
    Quick comparison of just the main architectures
    """
    results = []
    
    models = ['improved', 'multiscale', 'auxiliary']
    
    print("="*80)
    print("QUICK MODEL ARCHITECTURE COMPARISON")
    print("="*80)
    
    for model_class in models:
        print(f"\n{'='*80}")
        print(f"Testing: {model_class.upper()}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            model, val_acc, test_acc = train_with_gradual_unfreezing(
                model_class=model_class,
                num_epochs=20,  # Shorter for quick comparison
                use_focal_loss=True,
                use_mixup=True,
                use_tta=True,
                save_path=f'quick_{model_class}.pth'
            )
            
            training_time = time.time() - start_time
            
            results.append({
                'Model': model_class,
                'Val Accuracy': val_acc,
                'Test Accuracy': test_acc,
                'Training Time (min)': training_time / 60
            })
            
            print(f"\n✓ {model_class}: Val={val_acc:.2f}%, Test={test_acc:.2f}%")
            
        except Exception as e:
            print(f"\n✗ {model_class} failed: {str(e)}")
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Test Accuracy', ascending=False)
    
    print("\n" + "="*80)
    print("QUICK COMPARISON RESULTS")
    print("="*80)
    print(df_results.to_string(index=False))
    
    df_results.to_csv('quick_comparison_results.csv', index=False)
    
    return df_results


def ablation_study():
    """
    Ablation study to understand contribution of each component
    """
    results = []
    
    # Start with baseline and add components one by one
    configs = [
        (False, False, False, 'Baseline (no improvements)'),
        (True, False, False, '+ Focal Loss'),
        (True, True, False, '+ Focal Loss + Mixup'),
        (True, True, True, '+ Focal Loss + Mixup + TTA'),
    ]
    
    print("="*80)
    print("ABLATION STUDY - Understanding Component Contributions")
    print("="*80)
    
    for use_focal, use_mixup, use_tta, description in configs:
        print(f"\n{'='*80}")
        print(f"Testing: {description}")
        print(f"{'='*80}")
        
        try:
            model, val_acc, test_acc = train_with_gradual_unfreezing(
                model_class='improved',
                num_epochs=20,
                use_focal_loss=use_focal,
                use_mixup=use_mixup,
                use_tta=use_tta,
                save_path=f'ablation_{len(results)}.pth'
            )
            
            results.append({
                'Configuration': description,
                'Focal Loss': use_focal,
                'Mixup': use_mixup,
                'TTA': use_tta,
                'Val Accuracy': val_acc,
                'Test Accuracy': test_acc,
            })
            
            print(f"\n✓ {description}: Test={test_acc:.2f}%")
            
        except Exception as e:
            print(f"\n✗ Failed: {str(e)}")
    
    df_results = pd.DataFrame(results)
    
    # Calculate improvements
    baseline_acc = df_results.iloc[0]['Test Accuracy']
    df_results['Improvement (%)'] = df_results['Test Accuracy'] - baseline_acc
    
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(df_results.to_string(index=False))
    
    df_results.to_csv('ablation_study_results.csv', index=False)
    
    return df_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare different model configurations')
    parser.add_argument('--mode', type=str, default='quick', 
                       choices=['quick', 'full', 'ablation'],
                       help='Comparison mode: quick (3 models), full (all configs), ablation (component analysis)')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        print("Running quick comparison of main architectures...")
        results = quick_comparison()
    elif args.mode == 'full':
        print("Running full comparison of all configurations...")
        results = compare_all_models()
    elif args.mode == 'ablation':
        print("Running ablation study...")
        results = ablation_study()
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print(f"Results saved to CSV file")
