"""
BERTScore Evaluation for Ocular Diagnosis Comparison

This module uses BERTScore to evaluate the semantic similarity between
AI-generated ocular diagnoses and ground truth summaries.
BERTScore computes precision, recall, and F1 scores using contextual embeddings.
"""

import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
from bert_score import score
from transformers import AutoTokenizer




@dataclass
class BERTScoreResult:
    """Result of BERTScore evaluation for a single prediction"""
    clip_id: str
    precision: float
    recall: float
    f1: float
    ground_truth: str
    prediction: str


@dataclass
class BERTScoreSummary:
    """Summary statistics for BERTScore evaluation"""
    model_name: str
    num_samples: int
    mean_precision: float
    mean_recall: float
    mean_f1: float
    std_precision: float
    std_recall: float
    std_f1: float
    median_precision: float
    median_recall: float
    median_f1: float


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    lang: str = "en",
    model_type: str = "microsoft/deberta-xlarge-mnli",
    num_layers: Optional[int] = None,
    batch_size: int = 64,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute BERTScore for a batch of predictions against references.
    
    Args:
        predictions: List of predicted summaries
        references: List of ground truth summaries
        lang: Language code (default: "en")
        model_type: Pre-trained model to use for embeddings
                   Options: "microsoft/deberta-xlarge-mnli" (recommended),
                           "roberta-large", "bert-base-uncased", etc.
        num_layers: Number of layers to use (None = use all layers)
        batch_size: Batch size for processing
        verbose: Print progress information
        
    Returns:
        Tuple of (precision, recall, f1) numpy arrays
    """
    if len(predictions) != len(references):
        raise ValueError(f"Number of predictions ({len(predictions)}) must match references ({len(references)})")
    
    if len(predictions) == 0:
        raise ValueError("Cannot compute BERTScore on empty lists")
    
    # Truncate long texts to prevent tokenizer overflow (max 512 tokens ~ 1500 chars to be safe)
    # Also ensure no empty strings
    max_chars = 1500
    predictions_truncated = [p[:max_chars].strip() if p and len(p) > 0 else "N/A" for p in predictions]
    references_truncated = [r[:max_chars].strip() if r and len(r) > 0 else "N/A" for r in references]
    
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    tokenizer.model_max_length = 512

    # Compute BERTScore
    P, R, F1 = score(
        cands=predictions_truncated,
        refs=references_truncated,
        lang=lang,
        # model_type=model_type,
        # num_layers=num_layers,
        verbose=verbose,
        batch_size=batch_size,
        rescale_with_baseline=True  # Rescale scores using baseline,
    )
    
    # Convert to numpy arrays
    precision = P.numpy()
    recall = R.numpy()
    f1 = F1.numpy()
    
    return precision, recall, f1


def evaluate_single_prediction(
    clip_id: str,
    prediction: str,
    reference: str,
    lang: str = "en",
    model_type: str = "microsoft/deberta-xlarge-mnli"
) -> BERTScoreResult:
    """
    Evaluate a single prediction using BERTScore.
    
    Args:
        clip_id: Identifier for the video clip
        prediction: Predicted summary
        reference: Ground truth summary
        lang: Language code
        model_type: Pre-trained model to use
        
    Returns:
        BERTScoreResult with precision, recall, and F1 scores
    """
    P, R, F1 = compute_bertscore(
        predictions=[prediction],
        references=[reference],
        lang=lang,
        model_type=model_type,
        verbose=False
    )
    
    return BERTScoreResult(
        clip_id=clip_id,
        precision=float(P[0]),
        recall=float(R[0]),
        f1=float(F1[0]),
        ground_truth=reference,
        prediction=prediction
    )


def batch_evaluate_predictions(
    test_data: List[Dict],
    predictions: List[Dict],
    model_name: str,
    lang: str = "en",
    model_type: str = "microsoft/deberta-xlarge-mnli",
    batch_size: int = 64,
    output_file: Optional[str] = None,
    verbose: bool = True
) -> Tuple[List[BERTScoreResult], BERTScoreSummary]:
    """
    Batch evaluate predictions using BERTScore.
    
    Args:
        test_data: List of ground truth samples with 'clip_id' and 'summary'
        predictions: List of predictions with 'clip_id' and 'predicted_summary'
        model_name: Name of the model being evaluated
        lang: Language code
        model_type: Pre-trained model to use for BERTScore
        batch_size: Batch size for processing
        output_file: Optional path to save results as JSON
        verbose: Print progress information
        
    Returns:
        Tuple of (list of BERTScoreResult, BERTScoreSummary)
    """
    # Create lookup dictionaries
    gt_dict = {item['clip_id']: item for item in test_data}
    pred_dict = {item['clip_id']: item for item in predictions}
    
    # Find common clip_ids
    common_ids = sorted(set(gt_dict.keys()) & set(pred_dict.keys()))
    
    if len(common_ids) == 0:
        raise ValueError("No common clip_ids found between test data and predictions")
    
    if verbose:
        print(f"Evaluating {len(common_ids)} samples with BERTScore...")
        print(f"Model: {model_type}")
    
    # Prepare batched data
    clip_ids = []
    references = []
    candidates = []
    
    for clip_id in common_ids:
        clip_ids.append(clip_id)
        references.append(gt_dict[clip_id]['summary'])
        candidates.append(pred_dict[clip_id]['predicted_summary'])
    
    # Compute BERTScore for all samples
    precision, recall, f1 = compute_bertscore(
        predictions=candidates,
        references=references,
        lang=lang,
        model_type=model_type,
        batch_size=batch_size,
        verbose=verbose
    )
    
    # Create individual results
    results = []
    for i, clip_id in enumerate(clip_ids):
        result = BERTScoreResult(
            clip_id=clip_id,
            precision=float(precision[i]),
            recall=float(recall[i]),
            f1=float(f1[i]),
            ground_truth=references[i],
            prediction=candidates[i]
        )
        results.append(result)
    
    # Compute summary statistics
    summary = BERTScoreSummary(
        model_name=model_name,
        num_samples=len(results),
        mean_precision=float(np.mean(precision)),
        mean_recall=float(np.mean(recall)),
        mean_f1=float(np.mean(f1)),
        std_precision=float(np.std(precision)),
        std_recall=float(np.std(recall)),
        std_f1=float(np.std(f1)),
        median_precision=float(np.median(precision)),
        median_recall=float(np.median(recall)),
        median_f1=float(np.median(f1))
    )
    
    # Save results if output file specified
    if output_file:
        output_data = {
            'summary': asdict(summary),
            'results': [asdict(r) for r in results]
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        if verbose:
            print(f"\nResults saved to {output_file}")
    
    # Print summary statistics
    if verbose:
        print_summary_statistics(summary)
    
    return results, summary


def compare_two_models(
    test_data: List[Dict],
    predictions_1: List[Dict],
    predictions_2: List[Dict],
    model_1_name: str,
    model_2_name: str,
    lang: str = "en",
    model_type: str = "microsoft/deberta-xlarge-mnli",
    batch_size: int = 64,
    output_file: Optional[str] = None,
    verbose: bool = True
) -> Tuple[BERTScoreSummary, BERTScoreSummary]:
    """
    Compare two models using BERTScore.
    
    Args:
        test_data: List of ground truth samples
        predictions_1: Predictions from first model
        predictions_2: Predictions from second model
        model_1_name: Name of first model
        model_2_name: Name of second model
        lang: Language code
        model_type: Pre-trained model to use
        batch_size: Batch size for processing
        output_file: Optional path to save comparison results
        verbose: Print progress information
        
    Returns:
        Tuple of (summary_1, summary_2)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"BERTSCORE COMPARISON: {model_1_name} vs {model_2_name}")
        print(f"{'='*80}\n")
    
    # Evaluate first model
    if verbose:
        print(f"Evaluating {model_1_name}...")
    results_1, summary_1 = batch_evaluate_predictions(
        test_data=test_data,
        predictions=predictions_1,
        model_name=model_1_name,
        lang=lang,
        model_type=model_type,
        batch_size=batch_size,
        verbose=verbose
    )
    
    # Evaluate second model
    if verbose:
        print(f"\nEvaluating {model_2_name}...")
    results_2, summary_2 = batch_evaluate_predictions(
        test_data=test_data,
        predictions=predictions_2,
        model_name=model_2_name,
        lang=lang,
        model_type=model_type,
        batch_size=batch_size,
        verbose=verbose
    )
    
    # Print comparison
    if verbose:
        print_comparison_statistics(summary_1, summary_2)
    
    # Save comparison results
    if output_file:
        comparison_data = {
            'model_1': {
                'name': model_1_name,
                'summary': asdict(summary_1),
                'results': [asdict(r) for r in results_1]
            },
            'model_2': {
                'name': model_2_name,
                'summary': asdict(summary_2),
                'results': [asdict(r) for r in results_2]
            }
        }
        with open(output_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        if verbose:
            print(f"\nComparison results saved to {output_file}")
    
    return summary_1, summary_2


def print_summary_statistics(summary: BERTScoreSummary):
    """Print summary statistics for a single model."""
    print("\n" + "="*80)
    print(f"BERTSCORE SUMMARY: {summary.model_name}")
    print("="*80)
    print(f"\nNumber of Samples: {summary.num_samples}")
    print(f"\nPrecision:")
    print(f"  Mean:   {summary.mean_precision:.4f}")
    print(f"  Median: {summary.median_precision:.4f}")
    print(f"  Std:    {summary.std_precision:.4f}")
    print(f"\nRecall:")
    print(f"  Mean:   {summary.mean_recall:.4f}")
    print(f"  Median: {summary.median_recall:.4f}")
    print(f"  Std:    {summary.std_recall:.4f}")
    print(f"\nF1 Score:")
    print(f"  Mean:   {summary.mean_f1:.4f}")
    print(f"  Median: {summary.median_f1:.4f}")
    print(f"  Std:    {summary.std_f1:.4f}")
    print("="*80 + "\n")


def print_comparison_statistics(summary_1: BERTScoreSummary, summary_2: BERTScoreSummary):
    """Print comparison statistics between two models."""
    print("\n" + "="*80)
    print("BERTSCORE COMPARISON")
    print("="*80)
    print(f"\n{'Metric':<20} {summary_1.model_name:<25} {summary_2.model_name:<25} {'Difference':<15}")
    print("-" * 80)
    
    # Precision comparison
    prec_diff = summary_2.mean_precision - summary_1.mean_precision
    prec_symbol = "↑" if prec_diff > 0 else "↓" if prec_diff < 0 else "="
    print(f"{'Precision (mean)':<20} {summary_1.mean_precision:<25.4f} {summary_2.mean_precision:<25.4f} {prec_diff:>+.4f} {prec_symbol}")
    
    # Recall comparison
    rec_diff = summary_2.mean_recall - summary_1.mean_recall
    rec_symbol = "↑" if rec_diff > 0 else "↓" if rec_diff < 0 else "="
    print(f"{'Recall (mean)':<20} {summary_1.mean_recall:<25.4f} {summary_2.mean_recall:<25.4f} {rec_diff:>+.4f} {rec_symbol}")
    
    # F1 comparison
    f1_diff = summary_2.mean_f1 - summary_1.mean_f1
    f1_symbol = "↑" if f1_diff > 0 else "↓" if f1_diff < 0 else "="
    print(f"{'F1 Score (mean)':<20} {summary_1.mean_f1:<25.4f} {summary_2.mean_f1:<25.4f} {f1_diff:>+.4f} {f1_symbol}")
    
    print("-" * 80)
    
    # Determine winner
    if abs(f1_diff) < 0.001:
        winner = "Tie"
    elif f1_diff > 0:
        winner = summary_2.model_name
    else:
        winner = summary_1.model_name
    
    print(f"\nBest Model (by F1): {winner}")
    print("="*80 + "\n")


def load_csv_data(csv_path: str) -> List[Dict]:
    """Load CSV data into list of dictionaries."""
    import csv
    
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    return data


def normalize_prediction_data(predictions: List[Dict]) -> List[Dict]:
    """
    Normalize prediction data to standard format.
    
    Handles both formats:
    - filepath, pred -> clip_id, predicted_summary
    - clip_id, predicted_summary (already normalized)
    
    Args:
        predictions: List of prediction dictionaries
        
    Returns:
        Normalized list with 'clip_id' and 'predicted_summary' keys
    """
    import os
    
    normalized = []
    for pred in predictions:
        # Check if already in correct format
        if 'clip_id' in pred and 'predicted_summary' in pred:
            normalized.append(pred)
        # Handle filepath, pred format
        elif 'filepath' in pred and 'pred' in pred:
            # Extract clip_id from filepath (e.g., "clips/.../755384_00136.mp4" -> "755384_00136")
            filepath = pred['filepath']
            clip_id = os.path.splitext(os.path.basename(filepath))[0]
            normalized.append({
                'clip_id': clip_id,
                'predicted_summary': pred['pred']
            })
        else:
            raise ValueError(f"Unknown prediction format. Expected 'clip_id'/'predicted_summary' or 'filepath'/'pred', got: {pred.keys()}")
    
    return normalized


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate ocular diagnoses using BERTScore")
    parser.add_argument("--ground-truth", type=str, 
                       default="input/balanced_split_desc_test.csv",
                       help="Path to ground truth CSV file")
    parser.add_argument("--predictions-1", type=str, 
                       default="output/base_pred.csv",
                       help="Path to first model predictions CSV")
    parser.add_argument("--predictions-2", type=str, 
                       default=None,
                       help="Path to second model predictions CSV (optional, for comparison)")
    parser.add_argument("--model-1-name", type=str, default="Model 1",
                       help="Name of first model")
    parser.add_argument("--model-2-name", type=str, default="Model 2",
                       help="Name of second model (if comparing)")
    parser.add_argument("--output", type=str, default="output/bertscore_results.json",
                       help="Output JSON file path")
    parser.add_argument("--model-type", type=str, default="microsoft/deberta-xlarge-mnli",
                       help="Pre-trained model for BERTScore (default: microsoft/deberta-xlarge-mnli)")
    parser.add_argument("--lang", type=str, default="en",
                       help="Language code (default: en)")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for processing (default: 64)")
    parser.add_argument("--metric", type=str, default="f1",
                       choices=["precision", "recall", "f1"],
                       help="Primary metric to report (default: f1)")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    ground_truth = load_csv_data(args.ground_truth)
    predictions_1_raw = load_csv_data(args.predictions_1)
    
    # Normalize prediction formats
    print("Normalizing prediction formats...")
    predictions_1 = normalize_prediction_data(predictions_1_raw)
    
    # Check if comparing two models
    if args.predictions_2:
        predictions_2_raw = load_csv_data(args.predictions_2)
        predictions_2 = normalize_prediction_data(predictions_2_raw)
        
        # Run comparison
        summary_1, summary_2 = compare_two_models(
            test_data=ground_truth,
            predictions_1=predictions_1,
            predictions_2=predictions_2,
            model_1_name=args.model_1_name,
            model_2_name=args.model_2_name,
            lang=args.lang,
            model_type=args.model_type,
            batch_size=args.batch_size,
            output_file=args.output,
            verbose=True
        )
    else:
        # Evaluate single model
        results, summary = batch_evaluate_predictions(
            test_data=ground_truth,
            predictions=predictions_1,
            model_name=args.model_1_name,
            lang=args.lang,
            model_type=args.model_type,
            batch_size=args.batch_size,
            output_file=args.output,
            verbose=True
        )
    
    print(f"\nEvaluation complete! Results saved to {args.output}")
