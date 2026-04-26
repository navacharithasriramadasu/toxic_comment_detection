"""
Evaluate the Toxic Comment Detection Model and Save Results to File
Runs comprehensive evaluation and saves metrics to output files
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backend.model import ToxicityClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_dataset():
    """Create a comprehensive test dataset for evaluation"""
    
    test_data = [
        # Non-toxic comments
        ("This is a wonderful product, I love it!", [0,0,0,0,0,0]),
        ("Thank you so much for your help today", [0,0,0,0,0,0]),
        ("Great work on this project, keep it up!", [0,0,0,0,0,0]),
        ("I really appreciate your assistance", [0,0,0,0,0,0]),
        ("This is amazing and wonderful", [0,0,0,0,0,0]),
        ("Nice weather today, isn't it?", [0,0,0,0,0,0]),
        ("I learned a lot from this article", [0,0,0,0,0,0]),
        ("This is exactly what I was looking for", [0,0,0,0,0,0]),
        ("Excellent customer service", [0,0,0,0,0,0]),
        ("I would recommend this to everyone", [0,0,0,0,0,0]),
        
        # Toxic comments
        ("You are such an idiot and should die", [1,0,1,0,1,0]),
        ("Fuck you asshole piece of shit", [1,0,1,0,1,0]),
        ("I will kill you and your family", [1,0,0,1,0,0]),
        ("You stupid moron worthless human", [1,0,1,0,1,0]),
        ("Go to hell you disgusting pig", [1,0,1,0,1,0]),
        ("Worthless trash human being", [1,0,1,0,1,0]),
        ("I hate you and everyone like you", [1,0,0,0,1,0]),
        ("Disgusting pathetic loser", [1,0,1,0,1,0]),
        ("You should be banned forever", [1,0,0,0,1,0]),
        ("This is the worst thing ever", [1,0,0,0,1,0]),
        
        # Mildly toxic
        ("This is kind of stupid", [1,0,0,0,1,0]),
        ("I don't like this at all", [1,0,0,0,0,0]),
        ("This could be better", [0,0,0,0,0,0]),
        ("Not very good quality", [0,0,0,0,0,0]),
        ("I'm disappointed with this", [0,0,0,0,0,0]),
        
        # Severe toxic
        ("I will find you and hurt you", [1,1,0,1,1,0]),
        ("You deserve to suffer and die", [1,1,1,1,1,0]),
        ("All people like you should be eliminated", [1,1,0,1,1,1]),
        ("You are a disgrace to humanity", [1,1,1,0,1,0]),
        ("I hope terrible things happen to you", [1,1,0,1,1,0]),
        
        # Identity hate
        ("All [group] are terrible people", [1,0,0,0,0,1]),
        ("[group] should not be allowed here", [1,0,0,1,0,1]),
        ("I hate all [group] people", [1,0,0,0,1,1]),
        ("[group] are inferior to others", [1,0,0,0,0,1]),
        
        # Threats
        ("I know where you live", [1,0,0,1,0,0]),
        ("Watch your back", [1,0,0,1,0,0]),
        ("You're going to regret this", [1,0,0,1,0,0]),
        ("I'm coming for you", [1,0,0,1,0,0]),
    ]
    
    return test_data

def evaluate_model():
    """Run comprehensive model evaluation"""
    
    logger.info("Starting comprehensive model evaluation...")
    start_time = time.time()
    
    # Initialize classifier
    logger.info("Loading model...")
    classifier = ToxicityClassifier()
    
    # Create test dataset
    test_data = create_test_dataset()
    logger.info(f"Created test dataset with {len(test_data)} samples")
    
    # Run predictions
    logger.info("Running predictions...")
    predictions = []
    true_labels = []
    category_predictions = []
    category_true_labels = []
    
    category_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    
    for text, labels in test_data:
        true_labels.append(labels[0])  # Overall toxic label
        category_true_labels.append(labels)
        
        # Get prediction
        result = classifier.predict(text)
        predictions.append(1 if result["is_toxic"] else 0)
        
        # Get category predictions
        cat_preds = []
        for cat in category_names:
            cat_preds.append(1 if result["categories"][cat] >= 0.5 else 0)
        category_predictions.append(cat_preds)
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    category_predictions = np.array(category_predictions)
    category_true_labels = np.array(category_true_labels)
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics = {}
    
    # Overall metrics
    metrics["overall"] = {
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "precision": float(precision_score(true_labels, predictions, zero_division=0)),
        "recall": float(recall_score(true_labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(true_labels, predictions, zero_division=0)),
        "support": int(len(true_labels))
    }
    
    # Calculate ROC-AUC if possible
    try:
        # Get probability scores for ROC-AUC
        prob_scores = []
        for text, _ in test_data:
            result = classifier.predict(text)
            prob_scores.append(result["toxicity_score"])
        
        if len(np.unique(true_labels)) > 1:
            metrics["overall"]["roc_auc"] = float(roc_auc_score(true_labels, prob_scores))
        else:
            metrics["overall"]["roc_auc"] = 0.0
    except:
        metrics["overall"]["roc_auc"] = 0.0
    
    # Per-category metrics
    for i, cat_name in enumerate(category_names):
        y_true = category_true_labels[:, i]
        y_pred = category_predictions[:, i]
        
        if len(np.unique(y_true)) > 1:
            metrics[cat_name] = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
                "support": int(np.sum(y_true))
            }
        else:
            metrics[cat_name] = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "support": int(np.sum(y_true))
            }
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Detailed classification report
    try:
        class_report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)
        metrics["classification_report"] = class_report
    except:
        metrics["classification_report"] = {}
    
    # Error analysis
    errors = []
    for i, (text, true_label) in enumerate(test_data):
        pred_label = predictions[i]
        if true_label[0] != pred_label:
            errors.append({
                "text": text,
                "true_label": true_label[0],
                "predicted_label": int(pred_label),
                "error_type": "False Positive" if true_label[0] == 0 and pred_label == 1 else "False Negative"
            })
    
    # Create results dictionary
    results = {
        "evaluation_info": {
            "timestamp": datetime.now().isoformat(),
            "model_name": classifier.model_name,
            "model_mode": classifier.get_stats()["mode"],
            "test_samples": len(test_data),
            "evaluation_time_seconds": time.time() - start_time
        },
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "error_analysis": {
            "total_errors": len(errors),
            "error_rate": len(errors) / len(test_data),
            "false_positives": len([e for e in errors if e["error_type"] == "False Positive"]),
            "false_negatives": len([e for e in errors if e["error_type"] == "False Negative"]),
            "error_examples": errors[:5]  # First 5 errors
        },
        "model_stats": classifier.get_stats(),
        "test_data_summary": {
            "total_samples": len(test_data),
            "toxic_samples": sum(1 for _, labels in test_data if labels[0] == 1),
            "non_toxic_samples": sum(1 for _, labels in test_data if labels[0] == 0),
            "categories": category_names
        }
    }
    
    return results

def save_results(results):
    """Save evaluation results to files"""
    
    # Create output directory
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed JSON results
    json_file = output_dir / "model_evaluation_results.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Create human-readable report
    report_file = output_dir / "evaluation_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("TOXIC COMMENT DETECTION MODEL EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Basic info
        info = results["evaluation_info"]
        f.write(f"Evaluation Date: {info['timestamp']}\n")
        f.write(f"Model: {info['model_name']} ({info['model_mode']})\n")
        f.write(f"Test Samples: {info['test_samples']}\n")
        f.write(f"Evaluation Time: {info['evaluation_time_seconds']:.2f} seconds\n\n")
        
        # Overall metrics
        overall = results["metrics"]["overall"]
        f.write("OVERALL PERFORMANCE METRICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Accuracy: {overall['accuracy']:.4f}\n")
        f.write(f"Precision: {overall['precision']:.4f}\n")
        f.write(f"Recall: {overall['recall']:.4f}\n")
        f.write(f"F1 Score: {overall['f1_score']:.4f}\n")
        if 'roc_auc' in overall:
            f.write(f"ROC-AUC: {overall['roc_auc']:.4f}\n")
        f.write(f"Support: {overall['support']}\n\n")
        
        # Per-category metrics
        f.write("PER-CATEGORY PERFORMANCE\n")
        f.write("-" * 30 + "\n")
        for cat in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]:
            if cat in results["metrics"]:
                cat_metrics = results["metrics"][cat]
                f.write(f"{cat.replace('_', ' ').title()}:\n")
                f.write(f"  Accuracy: {cat_metrics['accuracy']:.4f}\n")
                f.write(f"  Precision: {cat_metrics['precision']:.4f}\n")
                f.write(f"  Recall: {cat_metrics['recall']:.4f}\n")
                f.write(f"  F1 Score: {cat_metrics['f1_score']:.4f}\n")
                f.write(f"  Support: {cat_metrics['support']}\n\n")
        
        # Confusion matrix
        cm = results["confusion_matrix"]
        f.write("CONFUSION MATRIX\n")
        f.write("-" * 20 + "\n")
        f.write("                Predicted\n")
        f.write("                Non-Toxic  Toxic\n")
        f.write("Actual Non-Toxic    {:8d}  {:5d}\n".format(cm[0][0], cm[0][1]))
        f.write("Actual Toxic        {:8d}  {:5d}\n\n".format(cm[1][0], cm[1][1]))
        
        # Error analysis
        errors = results["error_analysis"]
        f.write("ERROR ANALYSIS\n")
        f.write("-" * 15 + "\n")
        f.write(f"Total Errors: {errors['total_errors']}\n")
        f.write(f"Error Rate: {errors['error_rate']:.2%}\n")
        f.write(f"False Positives: {errors['false_positives']}\n")
        f.write(f"False Negatives: {errors['false_negatives']}\n\n")
        
        if errors["error_examples"]:
            f.write("Error Examples:\n")
            for i, error in enumerate(errors["error_examples"][:3]):
                f.write(f"{i+1}. {error['error_type']}: '{error['text'][:50]}...'\n")
                f.write(f"   True: {error['true_label']}, Predicted: {error['predicted_label']}\n")
        
        # Model stats
        stats = results["model_stats"]
        f.write("\nMODEL STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Analyzed: {stats['total_analyzed']}\n")
        f.write(f"Toxic Detected: {stats['toxic_detected']}\n")
        f.write(f"Non-Toxic: {stats['non_toxic']}\n")
        f.write(f"Toxic Rate: {stats['toxic_rate']:.2f}%\n")
    
    # Save summary metrics separately
    summary_file = output_dir / "summary_metrics.json"
    summary = {
        "timestamp": results["evaluation_info"]["timestamp"],
        "model": results["evaluation_info"]["model_name"],
        "test_samples": results["evaluation_info"]["test_samples"],
        "overall_metrics": results["metrics"]["overall"],
        "confusion_matrix": results["confusion_matrix"],
        "error_rate": results["error_analysis"]["error_rate"]
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"  - Detailed results: {json_file}")
    logger.info(f"  - Human readable report: {report_file}")
    logger.info(f"  - Summary metrics: {summary_file}")
    
    return output_dir

def main():
    """Main evaluation function"""
    try:
        logger.info("Starting model evaluation...")
        
        # Run evaluation
        results = evaluate_model()
        
        # Save results
        output_dir = save_results(results)
        
        # Print summary
        print("\n" + "=" * 60)
        print("MODEL EVALUATION COMPLETED")
        print("=" * 60)
        
        overall = results["metrics"]["overall"]
        print(f"Model: {results['evaluation_info']['model_name']}")
        print(f"Test Samples: {results['evaluation_info']['test_samples']}")
        print(f"Accuracy: {overall['accuracy']:.4f}")
        print(f"Precision: {overall['precision']:.4f}")
        print(f"Recall: {overall['recall']:.4f}")
        print(f"F1 Score: {overall['f1_score']:.4f}")
        if 'roc_auc' in overall:
            print(f"ROC-AUC: {overall['roc_auc']:.4f}")
        
        errors = results["error_analysis"]
        print(f"Error Rate: {errors['error_rate']:.2%}")
        print(f"False Positives: {errors['false_positives']}")
        print(f"False Negatives: {errors['false_negatives']}")
        
        print(f"\nResults saved to: {output_dir}")
        print("=" * 60)
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
