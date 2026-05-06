"""
Comparison utility for different evaluation methods.

This script helps compare results from:
1. LLM-as-Judge (GPT-4 based evaluation)
2. BERTScore (semantic similarity)
3. BERTScore Precision (hallucination detection)
"""

import json
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class EvaluationComparison:
    """Comparison of different evaluation methods"""
    clip_id: str
    ground_truth: str
    prediction: str
    
    # LLM-as-Judge results (if available)
    llm_winner: Optional[str] = None
    llm_hallucination_severity: Optional[str] = None
    llm_hallucination_count: Optional[int] = None
    
    # BERTScore results
    bertscore_precision: Optional[float] = None
    bertscore_recall: Optional[float] = None
    bertscore_f1: Optional[float] = None
    
    # BERTScore Precision results
    precision_category: Optional[str] = None


def load_llm_judge_results(json_path: str) -> Dict[str, Dict]:
    """Load LLM-as-Judge results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = {}
    for item in data:
        clip_id = item['clip_id']
        results[clip_id] = {
            'winner': item.get('winner'),
            'diagnosis_1_severity': item.get('diagnosis_1_analysis', {}).get('severity'),
            'diagnosis_1_hallucinations': len(item.get('diagnosis_1_analysis', {}).get('hallucinations', [])),
            'diagnosis_2_severity': item.get('diagnosis_2_analysis', {}).get('severity'),
            'diagnosis_2_hallucinations': len(item.get('diagnosis_2_analysis', {}).get('hallucinations', []))
        }
    
    return results


def load_bertscore_results(json_path: str) -> Dict[str, Dict]:
    """Load BERTScore results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = {}
    for item in data.get('results', []):
        clip_id = item['clip_id']
        results[clip_id] = {
            'precision': item['precision'],
            'recall': item['recall'],
            'f1': item['f1']
        }
    
    return results


def load_bertscore_precision_results(json_path: str) -> Dict[str, Dict]:
    """Load BERTScore Precision results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = {}
    for item in data.get('results', []):
        clip_id = item['clip_id']
        results[clip_id] = {
            'precision': item['precision'],
            'precision_category': item['precision_category']
        }
    
    return results


def analyze_correlation(
    llm_results: Optional[Dict] = None,
    bertscore_results: Optional[Dict] = None,
    bertscore_precision_results: Optional[Dict] = None,
    model_index: int = 1
) -> None:
    """
    Analyze correlation between different evaluation methods.
    
    Args:
        llm_results: LLM-as-Judge results
        bertscore_results: BERTScore results
        bertscore_precision_results: BERTScore Precision results
        model_index: Which model to analyze (1 or 2) for LLM results
    """
    if not any([llm_results, bertscore_results, bertscore_precision_results]):
        print("No results provided for correlation analysis")
        return
    
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    # Find common clip_ids
    all_clip_ids = set()
    if llm_results:
        all_clip_ids.update(llm_results.keys())
    if bertscore_results:
        all_clip_ids.update(bertscore_results.keys())
    if bertscore_precision_results:
        all_clip_ids.update(bertscore_precision_results.keys())
    
    # Analyze hallucination severity vs precision
    if llm_results and bertscore_precision_results:
        common_ids = set(llm_results.keys()) & set(bertscore_precision_results.keys())
        
        severity_key = f'diagnosis_{model_index}_severity'
        
        severity_to_precision = {
            'none': [],
            'minor': [],
            'moderate': [],
            'severe': []
        }
        
        for clip_id in common_ids:
            severity = llm_results[clip_id].get(severity_key)
            precision = bertscore_precision_results[clip_id]['precision']
            
            if severity in severity_to_precision:
                severity_to_precision[severity].append(precision)
        
        print("\nHallucination Severity vs BERTScore Precision:")
        print("-" * 80)
        print(f"{'Severity':<15} {'Count':<10} {'Mean Precision':<20} {'Std Precision':<15}")
        print("-" * 80)
        
        for severity in ['none', 'minor', 'moderate', 'severe']:
            precisions = severity_to_precision[severity]
            if precisions:
                mean_prec = np.mean(precisions)
                std_prec = np.std(precisions)
                print(f"{severity:<15} {len(precisions):<10} {mean_prec:<20.4f} {std_prec:<15.4f}")
        
        print("\nExpected Pattern:")
        print("  - 'none' severity → High precision (≥0.85)")
        print("  - 'minor' severity → Medium precision (0.70-0.85)")
        print("  - 'moderate/severe' → Low precision (<0.70)")
    
    # Analyze precision categories vs F1
    if bertscore_results and bertscore_precision_results:
        common_ids = set(bertscore_results.keys()) & set(bertscore_precision_results.keys())
        
        category_to_f1 = {
            'high': [],
            'medium': [],
            'low': []
        }
        
        for clip_id in common_ids:
            category = bertscore_precision_results[clip_id]['precision_category']
            f1 = bertscore_results[clip_id]['f1']
            
            if category in category_to_f1:
                category_to_f1[category].append(f1)
        
        print("\n\nPrecision Category vs BERTScore F1:")
        print("-" * 80)
        print(f"{'Category':<15} {'Count':<10} {'Mean F1':<20} {'Std F1':<15}")
        print("-" * 80)
        
        for category in ['high', 'medium', 'low']:
            f1_scores = category_to_f1[category]
            if f1_scores:
                mean_f1 = np.mean(f1_scores)
                std_f1 = np.std(f1_scores)
                print(f"{category:<15} {len(f1_scores):<10} {mean_f1:<20.4f} {std_f1:<15.4f}")
    
    print("="*80 + "\n")


def print_comparison_table(
    comparisons: List[EvaluationComparison],
    max_rows: int = 10
):
    """Print a comparison table of evaluation results."""
    print("\n" + "="*120)
    print("EVALUATION METHODS COMPARISON (Sample)")
    print("="*120)
    print(f"{'Clip ID':<15} {'LLM Severity':<15} {'LLM Halls':<10} {'BS Prec':<10} {'BS Recall':<10} {'BS F1':<10} {'Prec Cat':<12}")
    print("-" * 120)
    
    for i, comp in enumerate(comparisons[:max_rows]):
        llm_sev = comp.llm_hallucination_severity or "N/A"
        llm_halls = str(comp.llm_hallucination_count) if comp.llm_hallucination_count is not None else "N/A"
        bs_prec = f"{comp.bertscore_precision:.3f}" if comp.bertscore_precision is not None else "N/A"
        bs_recall = f"{comp.bertscore_recall:.3f}" if comp.bertscore_recall is not None else "N/A"
        bs_f1 = f"{comp.bertscore_f1:.3f}" if comp.bertscore_f1 is not None else "N/A"
        prec_cat = comp.precision_category or "N/A"
        
        print(f"{comp.clip_id:<15} {llm_sev:<15} {llm_halls:<10} {bs_prec:<10} {bs_recall:<10} {bs_f1:<10} {prec_cat:<12}")
    
    if len(comparisons) > max_rows:
        print(f"... ({len(comparisons) - max_rows} more rows)")
    
    print("="*120 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare results from different evaluation methods"
    )
    parser.add_argument("--llm-judge", type=str, default=None,
                       help="Path to LLM-as-Judge results JSON")
    parser.add_argument("--bertscore", type=str, default=None,
                       help="Path to BERTScore results JSON")
    parser.add_argument("--bertscore-precision", type=str, default=None,
                       help="Path to BERTScore Precision results JSON")
    parser.add_argument("--model-index", type=int, default=1, choices=[1, 2],
                       help="Which model to analyze from LLM-as-Judge (1 or 2)")
    parser.add_argument("--show-samples", type=int, default=10,
                       help="Number of sample rows to display")
    
    args = parser.parse_args()
    
    # Load results
    llm_results = None
    bertscore_results = None
    bertscore_precision_results = None
    
    if args.llm_judge:
        print(f"Loading LLM-as-Judge results from {args.llm_judge}...")
        llm_results = load_llm_judge_results(args.llm_judge)
        print(f"  Loaded {len(llm_results)} samples")
    
    if args.bertscore:
        print(f"Loading BERTScore results from {args.bertscore}...")
        bertscore_results = load_bertscore_results(args.bertscore)
        print(f"  Loaded {len(bertscore_results)} samples")
    
    if args.bertscore_precision:
        print(f"Loading BERTScore Precision results from {args.bertscore_precision}...")
        bertscore_precision_results = load_bertscore_precision_results(args.bertscore_precision)
        print(f"  Loaded {len(bertscore_precision_results)} samples")
    
    # Perform correlation analysis
    analyze_correlation(
        llm_results=llm_results,
        bertscore_results=bertscore_results,
        bertscore_precision_results=bertscore_precision_results,
        model_index=args.model_index
    )
    
    # Create comparison objects
    all_clip_ids = set()
    if llm_results:
        all_clip_ids.update(llm_results.keys())
    if bertscore_results:
        all_clip_ids.update(bertscore_results.keys())
    if bertscore_precision_results:
        all_clip_ids.update(bertscore_precision_results.keys())
    
    comparisons = []
    for clip_id in sorted(all_clip_ids):
        comp = EvaluationComparison(
            clip_id=clip_id,
            ground_truth="",
            prediction=""
        )
        
        if llm_results and clip_id in llm_results:
            severity_key = f'diagnosis_{args.model_index}_severity'
            halls_key = f'diagnosis_{args.model_index}_hallucinations'
            comp.llm_hallucination_severity = llm_results[clip_id].get(severity_key)
            comp.llm_hallucination_count = llm_results[clip_id].get(halls_key)
            comp.llm_winner = llm_results[clip_id].get('winner')
        
        if bertscore_results and clip_id in bertscore_results:
            comp.bertscore_precision = bertscore_results[clip_id]['precision']
            comp.bertscore_recall = bertscore_results[clip_id]['recall']
            comp.bertscore_f1 = bertscore_results[clip_id]['f1']
        
        if bertscore_precision_results and clip_id in bertscore_precision_results:
            comp.precision_category = bertscore_precision_results[clip_id]['precision_category']
            if comp.bertscore_precision is None:
                comp.bertscore_precision = bertscore_precision_results[clip_id]['precision']
        
        comparisons.append(comp)
    
    # Print comparison table
    if comparisons:
        print_comparison_table(comparisons, max_rows=args.show_samples)
    
    print("\nSummary:")
    print(f"  Total samples: {len(comparisons)}")
    if llm_results:
        print(f"  LLM-as-Judge samples: {len(llm_results)}")
    if bertscore_results:
        print(f"  BERTScore samples: {len(bertscore_results)}")
    if bertscore_precision_results:
        print(f"  BERTScore Precision samples: {len(bertscore_precision_results)}")


if __name__ == "__main__":
    main()
