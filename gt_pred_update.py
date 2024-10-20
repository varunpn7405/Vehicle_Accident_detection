import json

def load_data(ground_truth_file, predictions_file):
    with open(ground_truth_file) as f:
        ground_truth = json.load(f)

    with open(predictions_file) as f:
        predictions = json.load(f)

    return ground_truth, predictions

# Load data
ground_truth_file = 'img_gt.json'
predictions_file = 'img_pred.json'

ground_truth, predictions = load_data(ground_truth_file, predictions_file)

# Make a copy of the data
ground_truth_upd, predictions_upd = ground_truth.copy(), predictions.copy()

# Update the lists so they have the same length
for gt_key, pred_key in zip(ground_truth, predictions):
    
    # Get the annotations for the current key
    gt_annotations = ground_truth[gt_key]
    pred_annotations = predictions[pred_key]
    
    if len(gt_annotations) != len(pred_annotations):
        gt_len = len(gt_annotations)
        pred_len = len(pred_annotations)
        
        # Add padding to the smaller list
        if gt_len < pred_len:
            # Pad ground truth with empty boxes and None class
            for _ in range(pred_len - gt_len):
                ground_truth_upd[gt_key].append({"Bbox": [0, 0, 0, 0], "class": None})
        
        elif pred_len < gt_len:
            # Pad predictions with empty boxes and None class
            for _ in range(gt_len - pred_len):
                predictions_upd[pred_key].append({"Bbox": [0, 0, 0, 0], "class": None})

# Save updated data
with open("img_gt_upd.json", "w") as f:
    json.dump(ground_truth_upd, f, indent=4)

with open("img_pred_upd.json", "w") as f:
    json.dump(predictions_upd, f, indent=4)
