"""
Compare and visualize results from multi-class video classification benchmark.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path


def load_results(results_file):
    """Load benchmark results from JSON file"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results


def create_comparison_table(results):
    """Create comparison table of all models"""
    
    data = []
    for result in results:
        if result.get('status') == 'completed':
            data.append({
                'Model': result['model_name'].upper(),
                'Parameters': f"{result['num_params']:,}",
                'Diagnostic Acc (%)': f"{result['diagnostic_acc']:.2f}",
                'Diagnostic F1': f"{result['diagnostic_f1']:.3f}",
                'Subtype Acc (%)': f"{result['subtype_acc']:.2f}",
                'Subtype F1': f"{result['subtype_f1']:.3f}",
                'Training Time (min)': f"{result.get('training_time_minutes', 0):.1f}"
            })
    
    df = pd.DataFrame(data)
    return df


def plot_model_comparison(results, save_dir):
    """Create comparison plots"""
    
    completed_results = [r for r in results if r.get('status') == 'completed']
    
    if len(completed_results) == 0:
        print("No completed results to plot")
        return
    
    models = [r['model_name'].upper() for r in completed_results]
    diagnostic_acc = [r['diagnostic_acc'] for r in completed_results]
    diagnostic_f1 = [r['diagnostic_f1'] for r in completed_results]
    subtype_acc = [r['subtype_acc'] for r in completed_results]
    subtype_f1 = [r['subtype_f1'] for r in completed_results]
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Diagnostic Accuracy
    axes[0, 0].bar(models, diagnostic_acc, color='steelblue', alpha=0.7)
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Diagnostic Classification - Accuracy')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 100])
    for i, v in enumerate(diagnostic_acc):
        axes[0, 0].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom')
    
    # Diagnostic F1
    axes[0, 1].bar(models, diagnostic_f1, color='coral', alpha=0.7)
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('Diagnostic Classification - F1 Score')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim([0, 1])
    for i, v in enumerate(diagnostic_f1):
        axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # Subtype Accuracy
    axes[1, 0].bar(models, subtype_acc, color='mediumseagreen', alpha=0.7)
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Subtype Classification - Accuracy')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_ylim([0, 100])
    for i, v in enumerate(subtype_acc):
        axes[1, 0].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom')
    
    # Subtype F1
    axes[1, 1].bar(models, subtype_f1, color='mediumpurple', alpha=0.7)
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('Subtype Classification - F1 Score')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim([0, 1])
    for i, v in enumerate(subtype_f1):
        axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'model_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")
    plt.close()


def plot_accuracy_comparison(results, save_dir):
    """Create side-by-side accuracy comparison"""
    
    completed_results = [r for r in results if r.get('status') == 'completed']
    
    if len(completed_results) == 0:
        return
    
    models = [r['model_name'].upper() for r in completed_results]
    diagnostic_acc = [r['diagnostic_acc'] for r in completed_results]
    subtype_acc = [r['subtype_acc'] for r in completed_results]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, diagnostic_acc, width, label='Diagnostic', 
                   color='steelblue', alpha=0.7)
    bars2 = ax.bar(x + width/2, subtype_acc, width, label='Subtype',
                   color='coral', alpha=0.7)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'accuracy_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Accuracy comparison saved to: {save_path}")
    plt.close()


def generate_report(results, save_dir):
    """Generate markdown report"""
    
    report = []
    report.append("# Multi-Class Video Classification Benchmark Results\n")
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("\n## Summary\n")
    
    completed = [r for r in results if r.get('status') == 'completed']
    failed = [r for r in results if r.get('status') == 'failed']
    
    report.append(f"- Total models tested: {len(results)}\n")
    report.append(f"- Successfully completed: {len(completed)}\n")
    report.append(f"- Failed: {len(failed)}\n")
    
    if completed:
        report.append("\n## Model Performance\n")
        
        # Create table
        df = create_comparison_table(results)
        report.append("\n" + df.to_markdown(index=False) + "\n")
        
        # Best models
        report.append("\n## Best Performing Models\n")
        
        best_diagnostic = max(completed, key=lambda x: x['diagnostic_acc'])
        best_subtype = max(completed, key=lambda x: x['subtype_acc'])
        
        report.append(f"\n### Best Diagnostic Classification\n")
        report.append(f"- **Model**: {best_diagnostic['model_name'].upper()}\n")
        report.append(f"- **Accuracy**: {best_diagnostic['diagnostic_acc']:.2f}%\n")
        report.append(f"- **F1 Score**: {best_diagnostic['diagnostic_f1']:.3f}\n")
        report.append(f"- **Precision**: {best_diagnostic['diagnostic_precision']:.3f}\n")
        report.append(f"- **Recall**: {best_diagnostic['diagnostic_recall']:.3f}\n")
        
        report.append(f"\n### Best Subtype Classification\n")
        report.append(f"- **Model**: {best_subtype['model_name'].upper()}\n")
        report.append(f"- **Accuracy**: {best_subtype['subtype_acc']:.2f}%\n")
        report.append(f"- **F1 Score**: {best_subtype['subtype_f1']:.3f}\n")
        report.append(f"- **Precision**: {best_subtype['subtype_precision']:.3f}\n")
        report.append(f"- **Recall**: {best_subtype['subtype_recall']:.3f}\n")
    
    if failed:
        report.append("\n## Failed Models\n")
        for r in failed:
            report.append(f"- **{r['model_name'].upper()}**: {r.get('error', 'Unknown error')}\n")
    
    report_text = "".join(report)
    
    # Save report
    report_path = Path(save_dir) / 'BENCHMARK_REPORT.md'
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"\nBenchmark report saved to: {report_path}")
    
    return report_text


def main(args):
    results_file = Path(args.results_dir) / 'benchmark_results.json'
    
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        return
    
    print(f"Loading results from: {results_file}")
    results = load_results(results_file)
    
    print(f"\nFound {len(results)} model results")
    
    # Create comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE")
    print("="*80)
    df = create_comparison_table(results)
    print(df.to_string(index=False))
    
    # Save table to CSV
    csv_path = Path(args.results_dir) / 'comparison_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nTable saved to: {csv_path}")
    
    # Create plots
    print("\nGenerating comparison plots...")
    plot_model_comparison(results, args.results_dir)
    plot_accuracy_comparison(results, args.results_dir)
    
    # Generate report
    print("\nGenerating benchmark report...")
    report = generate_report(results, args.results_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare multi-class benchmark results')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory containing benchmark results')
    
    args = parser.parse_args()
    main(args)
