"""
LLM-as-Judge for Ocular Diagnosis Comparison

This module uses GPT-4 as a judge to compare two ocular diagnoses against ground truth,
identify which is more accurate, and detect hallucinations in each diagnosis.
"""

import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import openai
from openai import OpenAI


@dataclass
class HallucinationAnalysis:
    """Analysis of hallucinations in a diagnosis"""
    diagnosis_id: str  # e.g., "diagnosis_1" or "diagnosis_2"
    hallucinations: List[str]  # List of identified hallucinations
    severity: str  # "none", "minor", "moderate", "severe"
    explanation: str  # Detailed explanation of hallucinations
    

@dataclass
class ComparisonResult:
    """Result of comparing two diagnoses against ground truth"""
    clip_id: str
    ground_truth: Dict[str, str]
    diagnosis_1_analysis: HallucinationAnalysis
    diagnosis_2_analysis: HallucinationAnalysis
    winner: str  # "diagnosis_1", "diagnosis_2", or "tie"
    confidence: str  # "high", "medium", "low"
    reasoning: str  # Detailed reasoning for the decision
    overall_assessment: str  # Summary of comparison


def create_judge_prompt(
    ground_truth_summary: str,
    ground_truth_diagnosis: str,
    diagnosis_1_summary: str,
    diagnosis_1_name: str,
    diagnosis_2_summary: str,
    diagnosis_2_name: str
) -> str:
    """
    Create a detailed prompt for GPT to judge two diagnoses.
    
    Args:
        ground_truth_summary: The ground truth clinical summary
        ground_truth_diagnosis: The ground truth diagnosis labels
        diagnosis_1_summary: First diagnosis to evaluate
        diagnosis_1_name: Name/identifier for first diagnosis
        diagnosis_2_summary: Second diagnosis to evaluate
        diagnosis_2_name: Name/identifier for second diagnosis
        
    Returns:
        Formatted prompt for GPT judge
    """
    prompt = f"""You are an expert ophthalmologist and medical AI evaluator. Your task is to compare two AI-generated ocular ultrasound diagnoses against a ground truth diagnosis, identify hallucinations, and determine which is more accurate.

**GROUND TRUTH:**
Diagnosis Labels: {ground_truth_diagnosis}
Clinical Summary: {ground_truth_summary}

**DIAGNOSIS 1 ({diagnosis_1_name}):**
{diagnosis_1_summary}

**DIAGNOSIS 2 ({diagnosis_2_name}):**
{diagnosis_2_summary}

**YOUR TASK:**

1. **Identify Hallucinations in Diagnosis 1:**
   - List any factual claims that contradict the ground truth
   - Note any fabricated findings not supported by the ground truth
   - Identify incorrect anatomical descriptions
   - Flag any misclassifications (e.g., claiming RD when it's non-RD, or vice versa)
   - Assess severity: none, minor, moderate, or severe

2. **Identify Hallucinations in Diagnosis 2:**
   - Perform the same analysis as above

3. **Compare Overall Accuracy:**
   - Which diagnosis is closer to the ground truth?
   - Consider: diagnostic accuracy, anatomical correctness, clinical reasoning
   - Assign confidence level: high, medium, or low

4. **Provide Detailed Reasoning:**
   - Explain your decision with specific examples
   - Reference specific claims from each diagnosis

**OUTPUT FORMAT (JSON):**
```json
{{
  "diagnosis_1_hallucinations": {{
    "hallucinations": ["hallucination 1", "hallucination 2", ...],
    "severity": "none|minor|moderate|severe",
    "explanation": "detailed explanation of hallucinations found"
  }},
  "diagnosis_2_hallucinations": {{
    "hallucinations": ["hallucination 1", "hallucination 2", ...],
    "severity": "none|minor|moderate|severe",
    "explanation": "detailed explanation of hallucinations found"
  }},
  "winner": "diagnosis_1|diagnosis_2|tie",
  "confidence": "high|medium|low",
  "reasoning": "detailed reasoning for your decision",
  "overall_assessment": "brief summary of comparison"
}}
```

Provide ONLY the JSON output, no additional text."""
    
    return prompt


def parse_diagnosis_labels(diagnosis_text: str) -> Dict[str, str]:
    """
    Parse structured diagnosis text into components.
    
    Args:
        diagnosis_text: Diagnosis in format <diagnostic>X</diagnostic><subtype>Y</subtype>...
        
    Returns:
        Dictionary with diagnostic, subtype, and anatomical fields
    """
    import re
    
    result = {
        'diagnostic': 'unknown',
        'subtype': 'unknown',
        'anatomical': 'unknown'
    }
    
    # Extract diagnostic
    diag_match = re.search(r'<diagnostic>(.*?)</diagnostic>', diagnosis_text)
    if diag_match:
        result['diagnostic'] = diag_match.group(1)
    
    # Extract subtype
    subtype_match = re.search(r'<subtype>(.*?)</subtype>', diagnosis_text)
    if subtype_match:
        result['subtype'] = subtype_match.group(1)
    
    # Extract anatomical
    anat_match = re.search(r'<anatomical>(.*?)</anatomical>', diagnosis_text)
    if anat_match:
        result['anatomical'] = anat_match.group(1)
    
    return result


def compare_diagnoses_with_gpt(
    clip_id: str,
    ground_truth_summary: str,
    ground_truth_diagnosis: str,
    diagnosis_1_summary: str,
    diagnosis_1_name: str,
    diagnosis_2_summary: str,
    diagnosis_2_name: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    temperature: float = 0.1
) -> ComparisonResult:
    """
    Use GPT-4 as a judge to compare two diagnoses against ground truth.
    
    Args:
        clip_id: Identifier for the video clip
        ground_truth_summary: Ground truth clinical summary
        ground_truth_diagnosis: Ground truth diagnosis labels
        diagnosis_1_summary: First diagnosis to evaluate
        diagnosis_1_name: Name/identifier for first diagnosis (e.g., "Claude")
        diagnosis_2_summary: Second diagnosis to evaluate
        diagnosis_2_name: Name/identifier for second diagnosis (e.g., "GPT-4")
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        model: GPT model to use (default: gpt-4o)
        temperature: Sampling temperature (default: 0.1 for consistency)
        
    Returns:
        ComparisonResult with detailed analysis
    """
    # Initialize OpenAI client
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    client = OpenAI(api_key=api_key)
    
    # Create prompt
    prompt = create_judge_prompt(
        ground_truth_summary=ground_truth_summary,
        ground_truth_diagnosis=ground_truth_diagnosis,
        diagnosis_1_summary=diagnosis_1_summary,
        diagnosis_1_name=diagnosis_1_name,
        diagnosis_2_summary=diagnosis_2_summary,
        diagnosis_2_name=diagnosis_2_name
    )
    
    # Call GPT
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert ophthalmologist and medical AI evaluator. Provide responses in valid JSON format only."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        
        # Parse response
        result_json = json.loads(response.choices[0].message.content)
        
        # Create hallucination analyses
        diag1_analysis = HallucinationAnalysis(
            diagnosis_id="diagnosis_1",
            hallucinations=result_json["diagnosis_1_hallucinations"]["hallucinations"],
            severity=result_json["diagnosis_1_hallucinations"]["severity"],
            explanation=result_json["diagnosis_1_hallucinations"]["explanation"]
        )
        
        diag2_analysis = HallucinationAnalysis(
            diagnosis_id="diagnosis_2",
            hallucinations=result_json["diagnosis_2_hallucinations"]["hallucinations"],
            severity=result_json["diagnosis_2_hallucinations"]["severity"],
            explanation=result_json["diagnosis_2_hallucinations"]["explanation"]
        )
        
        # Parse ground truth
        gt_labels = parse_diagnosis_labels(ground_truth_diagnosis)
        
        # Create comparison result
        comparison = ComparisonResult(
            clip_id=clip_id,
            ground_truth=gt_labels,
            diagnosis_1_analysis=diag1_analysis,
            diagnosis_2_analysis=diag2_analysis,
            winner=result_json["winner"],
            confidence=result_json["confidence"],
            reasoning=result_json["reasoning"],
            overall_assessment=result_json["overall_assessment"]
        )
        
        return comparison
        
    except Exception as e:
        raise RuntimeError(f"Error calling GPT API: {str(e)}")


def batch_compare_diagnoses(
    test_data: List[Dict],
    predictions_1: List[Dict],
    predictions_2: List[Dict],
    model_1_name: str,
    model_2_name: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    output_file: Optional[str] = None
) -> List[ComparisonResult]:
    """
    Batch compare two sets of predictions against ground truth.
    
    Args:
        test_data: List of ground truth samples with 'clip_id', 'summary', 'diagnosis_text'
        predictions_1: List of predictions from first model with 'clip_id', 'predicted_summary'
        predictions_2: List of predictions from second model with 'clip_id', 'predicted_summary'
        model_1_name: Name of first model (e.g., "Claude")
        model_2_name: Name of second model (e.g., "GPT-4")
        api_key: OpenAI API key
        model: GPT model to use for judging
        output_file: Optional path to save results as JSON
        
    Returns:
        List of ComparisonResult objects
    """
    # Create lookup dictionaries
    gt_dict = {item['clip_id']: item for item in test_data}
    pred1_dict = {item['clip_id']: item for item in predictions_1}
    pred2_dict = {item['clip_id']: item for item in predictions_2}
    
    results = []
    
    # Find common clip_ids
    common_ids = set(gt_dict.keys()) & set(pred1_dict.keys()) & set(pred2_dict.keys())
    
    print(f"Comparing {len(common_ids)} samples...")
    
    for i, clip_id in enumerate(sorted(common_ids), 1):
        print(f"Processing {i}/{len(common_ids)}: {clip_id}")
        
        try:
            result = compare_diagnoses_with_gpt(
                clip_id=clip_id,
                ground_truth_summary=gt_dict[clip_id]['summary'],
                ground_truth_diagnosis=gt_dict[clip_id]['diagnosis_text'],
                diagnosis_1_summary=pred1_dict[clip_id]['predicted_summary'],
                diagnosis_1_name=model_1_name,
                diagnosis_2_summary=pred2_dict[clip_id]['predicted_summary'],
                diagnosis_2_name=model_2_name,
                api_key=api_key,
                model=model
            )
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {clip_id}: {str(e)}")
            continue
    
    # Save results if output file specified
    if output_file:
        results_dict = [asdict(r) for r in results]
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    # Print summary statistics
    print_summary_statistics(results, model_1_name, model_2_name)
    
    return results


def print_summary_statistics(
    results: List[ComparisonResult],
    model_1_name: str,
    model_2_name: str
):
    """Print summary statistics of comparison results."""
    
    total = len(results)
    model_1_wins = sum(1 for r in results if r.winner == "diagnosis_1")
    model_2_wins = sum(1 for r in results if r.winner == "diagnosis_2")
    ties = sum(1 for r in results if r.winner == "tie")
    
    # Hallucination severity counts
    model_1_severe = sum(1 for r in results if r.diagnosis_1_analysis.severity == "severe")
    model_1_moderate = sum(1 for r in results if r.diagnosis_1_analysis.severity == "moderate")
    model_1_minor = sum(1 for r in results if r.diagnosis_1_analysis.severity == "minor")
    model_1_none = sum(1 for r in results if r.diagnosis_1_analysis.severity == "none")
    
    model_2_severe = sum(1 for r in results if r.diagnosis_2_analysis.severity == "severe")
    model_2_moderate = sum(1 for r in results if r.diagnosis_2_analysis.severity == "moderate")
    model_2_minor = sum(1 for r in results if r.diagnosis_2_analysis.severity == "minor")
    model_2_none = sum(1 for r in results if r.diagnosis_2_analysis.severity == "none")
    
    # Average hallucination count
    avg_hall_1 = sum(len(r.diagnosis_1_analysis.hallucinations) for r in results) / total
    avg_hall_2 = sum(len(r.diagnosis_2_analysis.hallucinations) for r in results) / total
    
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\nTotal Comparisons: {total}")
    print(f"\n{model_1_name} Wins: {model_1_wins} ({model_1_wins/total*100:.1f}%)")
    print(f"{model_2_name} Wins: {model_2_wins} ({model_2_wins/total*100:.1f}%)")
    print(f"Ties: {ties} ({ties/total*100:.1f}%)")
    
    print(f"\n{model_1_name} Hallucination Severity:")
    print(f"  None: {model_1_none} ({model_1_none/total*100:.1f}%)")
    print(f"  Minor: {model_1_minor} ({model_1_minor/total*100:.1f}%)")
    print(f"  Moderate: {model_1_moderate} ({model_1_moderate/total*100:.1f}%)")
    print(f"  Severe: {model_1_severe} ({model_1_severe/total*100:.1f}%)")
    print(f"  Avg Hallucinations per Sample: {avg_hall_1:.2f}")
    
    print(f"\n{model_2_name} Hallucination Severity:")
    print(f"  None: {model_2_none} ({model_2_none/total*100:.1f}%)")
    print(f"  Minor: {model_2_minor} ({model_2_minor/total*100:.1f}%)")
    print(f"  Moderate: {model_2_moderate} ({model_2_moderate/total*100:.1f}%)")
    print(f"  Severe: {model_2_severe} ({model_2_severe/total*100:.1f}%)")
    print(f"  Avg Hallucinations per Sample: {avg_hall_2:.2f}")
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
    
    parser = argparse.ArgumentParser(description="Compare two ocular diagnoses using GPT as judge")
    parser.add_argument("--ground-truth", type=str, 
                       default="input/balanced_split_desc_test.csv",
                       help="Path to ground truth CSV file")
    parser.add_argument("--predictions-1", type=str, 
                       default="output/base_pred.csv",
                       help="Path to first model predictions CSV (default: base_pred.csv)")
    parser.add_argument("--predictions-2", type=str, 
                       default="output/FAVG_pred.csv",
                       help="Path to second model predictions CSV (default: FAVG_pred.csv)")
    parser.add_argument("--model-1-name", type=str, default="Base Model",
                       help="Name of first model")
    parser.add_argument("--model-2-name", type=str, default="FAVG Model",
                       help="Name of second model")
    parser.add_argument("--output", type=str, default="output/comparison_base_vs_FAVG.json",
                       help="Output JSON file path")
    parser.add_argument("--api-key", type=str, default=None,
                       help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                       help="GPT model to use for judging")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    ground_truth = load_csv_data(args.ground_truth)
    predictions_1_raw = load_csv_data(args.predictions_1)
    predictions_2_raw = load_csv_data(args.predictions_2)
    
    # Normalize prediction formats
    print("Normalizing prediction formats...")
    predictions_1 = normalize_prediction_data(predictions_1_raw)
    predictions_2 = normalize_prediction_data(predictions_2_raw)
    
    # Run comparison
    results = batch_compare_diagnoses(
        test_data=ground_truth,
        predictions_1=predictions_1,
        predictions_2=predictions_2,
        model_1_name=args.model_1_name,
        model_2_name=args.model_2_name,
        api_key=args.api_key,
        model=args.model,
        output_file=args.output
    )
    
    print(f"\nComparison complete! Results saved to {args.output}")
