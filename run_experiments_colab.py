"""
Colab-friendly experiment runner with progress tracking and memory monitoring
"""
import os
import sys
import torch
import subprocess
from datetime import datetime

# Set environment variables for memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

def check_environment():
    """Check and display environment information"""
    print("="*80)
    print("ENVIRONMENT CHECK")
    print("="*80)
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA Available")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  Free Memory: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        print("✗ CUDA Not Available - Will run on CPU (very slow!)")
    
    # Check PyTorch
    print(f"✓ PyTorch Version: {torch.__version__}")
    
    # Check Google Drive
    if os.path.exists('/content/drive'):
        print("✓ Google Drive mounted")
    else:
        print("⚠ Google Drive not mounted - results will be lost after session!")
    
    # Check data directory
    if os.path.exists('../erdes'):
        print("✓ Data directory found")
    else:
        print("✗ Data directory not found at ../erdes")
        return False
    
    print("="*80)
    print()
    return True


def setup_directories():
    """Create necessary directories"""
    save_dir = "/content/drive/MyDrive/EyeballProject/classifier_experiment"
    
    dirs = [
        save_dir,
        os.path.join(save_dir, "models"),
        os.path.join(save_dir, "checkpoints"),
        os.path.join(save_dir, "logs"),
        os.path.join(save_dir, "plots"),
        os.path.join(save_dir, "results"),
        os.path.join(save_dir, "explainability"),
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    print(f"✓ Directories created at: {save_dir}")
    print()
    return save_dir


def monitor_memory():
    """Display current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")


def run_experiments():
    """Run all experiments with monitoring"""
    print("="*80)
    print("STARTING EXPERIMENTS")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check environment
    if not check_environment():
        print("Environment check failed. Please fix issues before running.")
        return
    
    # Setup directories
    save_dir = setup_directories()
    
    # Initial memory check
    monitor_memory()
    print()
    
    # Import and run experiments
    print("Loading experiment modules...")
    try:
        from run_experiments import run_all_experiments
        
        print("Starting experiment execution...")
        print("This will take several hours. You can monitor progress in the logs.")
        print()
        
        results = run_all_experiments()
        
        print()
        print("="*80)
        print("EXPERIMENTS COMPLETED!")
        print("="*80)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Summary
        completed = sum(1 for r in results if r.get('status') == 'completed')
        failed = sum(1 for r in results if r.get('status') == 'failed')
        
        print(f"Total Experiments: {len(results)}")
        print(f"Completed: {completed}")
        print(f"Failed: {failed}")
        print()
        
        if completed > 0:
            print("Best Results:")
            completed_results = [r for r in results if r.get('status') == 'completed']
            best = max(completed_results, key=lambda x: x.get('test_acc', 0))
            print(f"  Experiment: {best['experiment_name']}")
            print(f"  Test Accuracy: {best['test_acc']:.2f}%")
            print(f"  Test F1: {best['test_f1']:.3f}")
            print(f"  Test AUC: {best['test_auc']:.3f}")
        
        print()
        print(f"Results saved to: {save_dir}")
        print()
        print("Next Steps:")
        print(f"1. Review: {os.path.join(save_dir, 'EXPERIMENT_REPORT.md')}")
        print(f"2. Check: {os.path.join(save_dir, 'results/experiment_summary.csv')}")
        print("3. Run visualize_explainability.py for interpretability analysis")
        print()
        
        # Final memory check
        monitor_memory()
        
    except Exception as e:
        print()
        print("="*80)
        print("ERROR OCCURRED!")
        print("="*80)
        print(f"Error: {str(e)}")
        print()
        import traceback
        traceback.print_exc()
        print()
        print(f"Check logs in: {save_dir}/logs/")


def run_single_experiment(exp_name):
    """Run a single experiment by name"""
    print(f"Running single experiment: {exp_name}")
    print()
    
    if not check_environment():
        return
    
    save_dir = setup_directories()
    
    from run_experiments import run_single_experiment, ExperimentLogger
    
    # Find experiment config
    from run_experiments import run_all_experiments
    # This is a bit hacky but works
    import run_experiments as re_module
    experiments = re_module.run_all_experiments.__code__.co_consts
    
    # For now, just run all and filter
    print("Note: To run a single experiment, modify run_experiments.py")
    print("or use the experiment index.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ocular ultrasound classifier experiments')
    parser.add_argument('--experiment', type=str, default=None, 
                       help='Run specific experiment by name (e.g., exp08_explainable_lower_dropout)')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check environment, do not run experiments')
    
    args = parser.parse_args()
    
    if args.check_only:
        check_environment()
        setup_directories()
    elif args.experiment:
        run_single_experiment(args.experiment)
    else:
        run_experiments()
