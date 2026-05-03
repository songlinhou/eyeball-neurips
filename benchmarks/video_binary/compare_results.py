import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse


def load_results(results_file):
    """Load benchmark results from JSON file"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results


def create_comparison_table(results):
    """Create a comparison table of all models"""
    data = []
    for result in results:
        if result.get('status') == 'completed':
            data.append({
                'Model': result['model_name'].upper(),
                'Accuracy (%)': f"{result['test_acc']:.2f}",
                'Precision': f"{result['test_precision']:.3f}",
                'Recall': f"{result['test_recall']:.3f}",
                'F1 Score': f"{result['test_f1']:.3f}",
                'AUC': f"{result['test_auc']:.3f}",
                'Parameters (M)': f"{result['num_params'] / 1e6:.2f}",
                'Training Time (min)': f"{result['training_time_minutes']:.1f}"
            })
    
    df = pd.DataFrame(data)
    return df


def plot_metrics_comparison(results, save_dir):
    """Plot comparison of key metrics across models"""
    completed_results = [r for r in results if r.get('status') == 'completed']
    
    if not completed_results:
        print("No completed results to plot")
        return
    
    models = [r['model_name'].upper() for r in completed_results]
    accuracies = [r['test_acc'] for r in completed_results]
    precisions = [r['test_precision'] * 100 for r in completed_results]
    recalls = [r['test_recall'] * 100 for r in completed_results]
    f1_scores = [r['test_f1'] * 100 for r in completed_results]
    aucs = [r['test_auc'] * 100 for r in completed_results]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    axes[0, 0].bar(models, accuracies, color=colors)
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Test Accuracy Comparison')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    
    axes[0, 1].bar(models, precisions, color=colors)
    axes[0, 1].set_ylabel('Precision (%)')
    axes[0, 1].set_title('Test Precision Comparison')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(precisions):
        axes[0, 1].text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    
    axes[0, 2].bar(models, recalls, color=colors)
    axes[0, 2].set_ylabel('Recall (%)')
    axes[0, 2].set_title('Test Recall Comparison')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(axis='y', alpha=0.3)
    for i, v in enumerate(recalls):
        axes[0, 2].text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    
    axes[1, 0].bar(models, f1_scores, color=colors)
    axes[1, 0].set_ylabel('F1 Score (%)')
    axes[1, 0].set_title('Test F1 Score Comparison')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(f1_scores):
        axes[1, 0].text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    
    axes[1, 1].bar(models, aucs, color=colors)
    axes[1, 1].set_ylabel('AUC (%)')
    axes[1, 1].set_title('Test AUC Comparison')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(aucs):
        axes[1, 1].text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    
    params = [r['num_params'] / 1e6 for r in completed_results]
    times = [r['training_time_minutes'] for r in completed_results]
    
    ax2 = axes[1, 2]
    ax2.scatter(params, accuracies, s=100, c=colors, alpha=0.6)
    for i, model in enumerate(models):
        ax2.annotate(model, (params[i], accuracies[i]), fontsize=8, ha='right')
    ax2.set_xlabel('Parameters (M)')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy vs Model Size')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'metrics_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved metrics comparison plot to: {plot_path}")
    plt.close()


def plot_confusion_matrices(results, save_dir):
    """Plot confusion matrices for all models"""
    completed_results = [r for r in results if r.get('status') == 'completed' and 'confusion_matrix' in r]
    
    if not completed_results:
        print("No confusion matrices to plot")
        return
    
    n_models = len(completed_results)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, result in enumerate(completed_results):
        cm = np.array(result['confusion_matrix'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Intact', 'Detached'],
                   yticklabels=['Intact', 'Detached'])
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
        axes[idx].set_title(f"{result['model_name'].upper()}\nAcc: {result['test_acc']:.2f}%, F1: {result['test_f1']:.3f}")
    
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'confusion_matrices.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrices plot to: {plot_path}")
    plt.close()


def plot_efficiency_analysis(results, save_dir):
    """Plot efficiency analysis: accuracy vs parameters and training time"""
    completed_results = [r for r in results if r.get('status') == 'completed']
    
    if not completed_results:
        print("No completed results for efficiency analysis")
        return
    
    models = [r['model_name'].upper() for r in completed_results]
    accuracies = [r['test_acc'] for r in completed_results]
    f1_scores = [r['test_f1'] * 100 for r in completed_results]
    params = [r['num_params'] / 1e6 for r in completed_results]
    times = [r['training_time_minutes'] for r in completed_results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    axes[0].scatter(params, accuracies, s=200, c=colors, alpha=0.6, edgecolors='black', linewidth=1.5)
    for i, model in enumerate(models):
        axes[0].annotate(model, (params[i], accuracies[i]), fontsize=10, ha='center', va='bottom')
    axes[0].set_xlabel('Model Parameters (Millions)', fontsize=12)
    axes[0].set_ylabel('Test Accuracy (%)', fontsize=12)
    axes[0].set_title('Model Efficiency: Accuracy vs Parameters', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(times, f1_scores, s=200, c=colors, alpha=0.6, edgecolors='black', linewidth=1.5)
    for i, model in enumerate(models):
        axes[1].annotate(model, (times[i], f1_scores[i]), fontsize=10, ha='center', va='bottom')
    axes[1].set_xlabel('Training Time (minutes)', fontsize=12)
    axes[1].set_ylabel('Test F1 Score (%)', fontsize=12)
    axes[1].set_title('Training Efficiency: F1 Score vs Time', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'efficiency_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved efficiency analysis plot to: {plot_path}")
    plt.close()


def plot_radar_chart(results, save_dir):
    """Create radar chart comparing models across multiple metrics"""
    completed_results = [r for r in results if r.get('status') == 'completed']
    
    if not completed_results:
        print("No completed results for radar chart")
        return
    
    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(completed_results)))
    
    for idx, result in enumerate(completed_results):
        values = [
            result['test_acc'],
            result['test_precision'] * 100,
            result['test_recall'] * 100,
            result['test_f1'] * 100,
            result['test_auc'] * 100
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=result['model_name'].upper(), color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], size=10)
    ax.grid(True)
    ax.set_title('Model Performance Comparison (Radar Chart)', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'radar_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved radar chart to: {plot_path}")
    plt.close()


def generate_report(results, save_dir):
    """Generate a comprehensive markdown report"""
    report_path = os.path.join(save_dir, 'BENCHMARK_REPORT.md')
    
    with open(report_path, 'w') as f:
        f.write("# Video Classification Benchmark Report\n\n")
        f.write("## Overview\n\n")
        f.write("This report presents a comprehensive comparison of state-of-the-art video classification methods ")
        f.write("for medical video analysis on the ERDES dataset (macula detached vs intact classification).\n\n")
        
        f.write("## Models Evaluated\n\n")
        f.write("The following models were evaluated:\n\n")
        
        model_descriptions = {
            'resnet3d': 'ResNet3D - Standard 3D ResNet baseline',
            'i3d': 'I3D - Inflated 3D ConvNet (CVPR 2017)',
            'slowfast': 'SlowFast - Dual-pathway network for video recognition (ICCV 2019)',
            'x3d': 'X3D - Efficient video network with progressive expansion (CVPR 2020)',
            'mvit': 'MViT - Multiscale Vision Transformer (ICCV 2021)',
            'videomae': 'VideoMAE - Masked Autoencoder for video (NeurIPS 2022)',
            'timesformer': 'TimeSformer - Space-time attention transformer (ICML 2021)',
            'c3d': 'C3D - Classic 3D CNN baseline (ICCV 2015)'
        }
        
        for result in results:
            model_name = result['model_name']
            status = result.get('status', 'unknown')
            desc = model_descriptions.get(model_name, 'Video classification model')
            f.write(f"- **{model_name.upper()}**: {desc} - Status: {status}\n")
        
        f.write("\n## Results Summary\n\n")
        
        df = create_comparison_table(results)
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        
        completed_results = [r for r in results if r.get('status') == 'completed']
        if completed_results:
            best_acc = max(completed_results, key=lambda x: x['test_acc'])
            best_f1 = max(completed_results, key=lambda x: x['test_f1'])
            best_auc = max(completed_results, key=lambda x: x['test_auc'])
            
            f.write("## Best Performing Models\n\n")
            f.write(f"- **Best Accuracy**: {best_acc['model_name'].upper()} ({best_acc['test_acc']:.2f}%)\n")
            f.write(f"- **Best F1 Score**: {best_f1['model_name'].upper()} ({best_f1['test_f1']:.3f})\n")
            f.write(f"- **Best AUC**: {best_auc['model_name'].upper()} ({best_auc['test_auc']:.3f})\n\n")
            
            f.write("## Key Findings\n\n")
            f.write("### Performance Analysis\n\n")
            
            avg_acc = np.mean([r['test_acc'] for r in completed_results])
            avg_f1 = np.mean([r['test_f1'] for r in completed_results])
            
            f.write(f"- Average test accuracy across all models: {avg_acc:.2f}%\n")
            f.write(f"- Average F1 score across all models: {avg_f1:.3f}\n")
            f.write(f"- Performance range: {min([r['test_acc'] for r in completed_results]):.2f}% - {max([r['test_acc'] for r in completed_results]):.2f}%\n\n")
            
            f.write("### Efficiency Analysis\n\n")
            
            params_list = [r['num_params'] / 1e6 for r in completed_results]
            times_list = [r['training_time_minutes'] for r in completed_results]
            
            f.write(f"- Model size range: {min(params_list):.2f}M - {max(params_list):.2f}M parameters\n")
            f.write(f"- Training time range: {min(times_list):.1f} - {max(times_list):.1f} minutes\n\n")
            
            most_efficient = min(completed_results, key=lambda x: x['num_params'] / (x['test_acc'] + 1e-6))
            f.write(f"- Most parameter-efficient model: {most_efficient['model_name'].upper()} ")
            f.write(f"({most_efficient['num_params']/1e6:.2f}M params, {most_efficient['test_acc']:.2f}% acc)\n\n")
        
        f.write("## Visualizations\n\n")
        f.write("The following visualizations are available in the results directory:\n\n")
        f.write("- `metrics_comparison.png` - Comparison of all metrics across models\n")
        f.write("- `confusion_matrices.png` - Confusion matrices for all models\n")
        f.write("- `efficiency_analysis.png` - Accuracy vs parameters and F1 vs training time\n")
        f.write("- `radar_comparison.png` - Radar chart comparing all metrics\n\n")
        
        f.write("## Recommendations\n\n")
        if completed_results:
            f.write("Based on the benchmark results:\n\n")
            f.write(f"1. **For best accuracy**: Use {best_acc['model_name'].upper()}\n")
            f.write(f"2. **For balanced performance**: Use {best_f1['model_name'].upper()}\n")
            f.write(f"3. **For efficiency**: Consider model size and training time trade-offs\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("This benchmark provides a comprehensive comparison of state-of-the-art video classification methods ")
        f.write("for medical video analysis. The results can guide model selection based on specific requirements ")
        f.write("for accuracy, efficiency, and computational resources.\n")
    
    print(f"Saved benchmark report to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare video classification benchmark results')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory containing benchmark results')
    parser.add_argument('--results_file', type=str, default='benchmark_results.json',
                       help='Name of results JSON file')
    
    args = parser.parse_args()
    
    results_path = os.path.join(args.results_dir, args.results_file)
    
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
        return
    
    print(f"Loading results from: {results_path}")
    results = load_results(results_path)
    
    print(f"\nFound {len(results)} model results")
    completed = sum(1 for r in results if r.get('status') == 'completed')
    print(f"Completed: {completed}, Failed: {len(results) - completed}")
    
    print("\n" + "="*80)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*80 + "\n")
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    print("Creating comparison table...")
    df = create_comparison_table(results)
    print("\n" + df.to_string(index=False))
    
    csv_path = os.path.join(args.results_dir, 'comparison_table.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved comparison table to: {csv_path}")
    
    print("\nGenerating plots...")
    plot_metrics_comparison(results, args.results_dir)
    plot_confusion_matrices(results, args.results_dir)
    plot_efficiency_analysis(results, args.results_dir)
    plot_radar_chart(results, args.results_dir)
    
    print("\nGenerating comprehensive report...")
    generate_report(results, args.results_dir)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
