import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def load_data(ground_truth_file, predictions_file):
    with open(ground_truth_file) as f:
        ground_truth = json.load(f)

    with open(predictions_file) as f:
        predictions = json.load(f)

    return ground_truth, predictions

def iou(box1, box2):
    # Calculate the intersection coordinates
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    # Calculate the area of intersection
    intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Calculate the area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the area of union
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0

def calculate_metrics(ground_truth, predictions, iou_threshold=0.5):
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives

    # For classification report
    y_true = []  # Ground truth classes
    y_pred = []  # Predicted classes

    for image in ground_truth:
        gt_boxes = ground_truth[image]
        pred_boxes = predictions[image]

        matched_gt = [False] * len(gt_boxes)  # Track which ground truths have been matched

        for pred in pred_boxes:
            pred_box = pred['Bbox']
            pred_class = pred['class']
              # Append predicted class for report

            best_iou = 0
            best_gt_idx = -1

            for idx, gt in enumerate(gt_boxes):
                gt_box = gt['Bbox']
                gt_class = gt['class']

                # Only consider ground truths that match the predicted class
                if gt_class == pred_class and not matched_gt[idx]:
                    current_iou = iou(pred_box, gt_box)
                    if current_iou > best_iou:
                        best_iou = current_iou
                        best_gt_idx = idx

            # Check if the best IoU exceeds the threshold
            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp += 1  # Count as true positive
                matched_gt[best_gt_idx] = True  # Mark this ground truth as matched
            else:
                fp += 1  # Count as false positive

        fn += matched_gt.count(False)  # Count unmatched ground truths as false negatives

        # Append ground truth classes for the report
        y_true.extend(gt['class'] for gt in gt_boxes)
        y_pred.extend(pred['class'] for pred in pred_boxes)

        y_true = [label if label is not None else -1 for label in y_true]
        y_pred = [label if label is not None else -1 for label in y_pred]

    # Calculate precision, recall, F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return tp, fp, fn, precision, recall, f1_score, y_true, y_pred

def plot_metrics(precision, recall, f1_score):
    metrics = [precision, recall, f1_score]
    labels = ['Precision', 'Recall', 'F1 Score']
    
    plt.figure(figsize=(8, 5))
    plt.bar(labels, metrics, color=['blue', 'orange', 'green'])
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('Performance Metrics')
    plt.grid(axis='y')

    for i, v in enumerate(metrics):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')

    plt.show()

def main(ground_truth_file, predictions_file):
    ground_truth, predictions = load_data(ground_truth_file, predictions_file)

    tp, fp, fn, precision, recall, f1_score, y_true, y_pred = calculate_metrics(ground_truth, predictions)

    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")

    # Generate classification report
    print("\nClassification Report:")
    print(f"Length of y_true: {len(y_true)}")
    print(f"Length of y_pred: {len(y_pred)}")
    print(classification_report(y_true, y_pred))

    # Plot metrics
    plot_metrics(precision, recall, f1_score)

# Example usage
ground_truth_file = 'img_gt_upd.json'
predictions_file = 'img_pred_upd.json'
main(ground_truth_file, predictions_file)
