from pathlib import Path
from compound import ComEM
import pandas as pd
from sklearn.metrics import classification_report

results = {}
dataset_files = sorted(Path("data/llm4em").glob("*.csv"))
compound = ComEM(use_cache=False)

def create_subsets(instances):
    """Create two lists of at most 15 instances each: one list with more 1s, one list with more 0s"""
    more_ones = []
    more_zeros = []
    
    for instance in instances:
        if len(instance['labels']) < 10:
            continue

        num_zeros = instance['labels'].count(0)
        num_ones = instance['labels'].count(1)
        
        if num_ones > 0 and num_zeros > num_ones and len(more_zeros) < 15:
            more_zeros.append(instance)
        elif num_zeros > 0 and num_ones > num_zeros and len(more_ones) < 15:
            more_ones.append(instance)
            
        if len(more_zeros) == 15 and len(more_ones) == 15:
            break

    return more_ones+more_zeros

# Find best K value with tiebreakers
def get_best_k(k_results):
    best_k = None
    best_metrics = (-1, -1, -1)  # (f1, recall, precision)
    
    for k, metrics in k_results.items():
        current_metrics = (metrics['f1'], metrics['recall'], metrics['precision'])
        if current_metrics > best_metrics:  # Tuple comparison will check in order
            best_metrics = current_metrics
            best_k = k
        elif current_metrics == best_metrics: # always pick higher k if equal
            best_k = max(best_k, k)
    
    return best_k, best_metrics

for file in dataset_files:
    dataset = file.stem
    print(f"[bold magenta]{dataset}[/bold magenta]")
    df = pd.read_csv(file)

    # Create instances list
    groupby = list(
        df.groupby("id_left")[["record_left", "record_right", "label"]]
        .apply(lambda x: x.to_dict("list"))
        .to_dict()
        .items()
    )

    instances = create_subsets([
        {
            "anchor": v["record_left"][0],
            "candidates": v["record_right"],
            "labels": v["label"],
        }
        for _, v in groupby
    ])

    print('Intance Length: ', len(instances))

    # Store results for each K value
    k_results = {}
    
    # Test different K values
    for k in range(6, 11):
        preds_lst = [compound(it, topK=k) for it in instances]
        preds = [pred for preds in preds_lst for pred in preds]
        labels = [label for it in instances for label in it["labels"]]
        
        # Get classification report
        report = classification_report(labels[: len(preds)], preds, digits=4, output_dict=True)
        
        # Store all metrics for this K
        k_results[k] = {
            'f1': report['weighted avg']['f1-score'],
            'recall': report['weighted avg']['recall'],
            'precision': report['weighted avg']['precision']
        }

        print(f'K {k}', k_results[k])

    best_k, best_metrics = get_best_k(k_results)
    print(f"\nBest K value for {dataset}: {best_k}")
    print(f"Metrics - F1: {best_metrics[0]:.4f}, Recall: {best_metrics[1]:.4f}, Precision: {best_metrics[2]:.4f}")
