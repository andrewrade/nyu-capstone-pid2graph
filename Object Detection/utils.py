
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ultralytics.utils.metrics import DetMetrics, ConfusionMatrix
from sahi.predict import get_sliced_prediction
from ultralytics.utils import metrics

def plot_bboxes(image, boxes, box_color, title):
    """
    Plots bounding boxes on the image.
    
    Parameters:
    - image: The background image on which to plot the boxes
    - boxes: Bounding boxes in xyxy format
    - box_color: The color for the bounding boxes
    - title: Title of the plot
    """
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)
    
    ax.set_title(title)
    plt.show()


def detection_metrics(iou, iou_threshold=0.7):    
    
    valid_iou_mask = iou >= iou_threshold
    overall_iou = iou[valid_iou_mask].mean() if valid_iou_mask.any() else torch.tensor(0.0)
    
    # Determine true positives, false positives, and false negatives
    true_positives = (valid_iou_mask.sum(dim=1) > 0).sum().item()  # Number of predictions with at least one valid IoU
    false_positives = (valid_iou_mask.sum(dim=0) == 0).sum().item()  # Predictions without a valid IoU
    false_negatives = (valid_iou_mask.sum(dim=1) == 0).sum().item()  # Ground truths without any valid predictions

    # Calculate precision and recall
    #precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    #recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    #print(f"Overall IoU: {overall_iou.item():.3f}")
    #print(f"Precision: {precision:.4f}")
    #print(f"Recall: {recall:.4f}")
    return overall_iou, true_positives, false_positives, false_negatives


def convert_yolo_to_torch(coords, height, width, x1, y1, cropped=None):
        """
        Converts yolo format (class_id   center_x  center_y  width  height) -->
        to torch format (x1, y1, x2, y2). Yolo format is normalized & torch format 
        isn't, so need to scale by image dimensions
        """
        #label, center_x, center_y, bb_width, bb_height = coords

        # Bounding Boxes annotated on uncropped image, adjust to orig dims
        coords[:, 1] *= width 
        coords[:, 2] *= height 
        coords[:, 3] *= width 
        coords[:, 4] *= height 

        if cropped:
            coords[:, 1] -= x1
            coords[:, 2] -= y1 

        x_min = coords[:, 1] - (0.5 * coords[:, 3])
        y_min = coords[:, 2] - (0.5 * coords[:, 4])

        x_max = coords[:, 1] + (0.5 * coords[:, 3])
        y_max = coords[:, 2] + (0.5 * coords[:, 4])
        
        return torch.stack((x_min, y_min, x_max, y_max), dim=1)

def sliced_inference(model, img_path, slice_height, slice_width, h_ratio, w_ratio):
    preds = get_sliced_prediction(
        img_path,
        model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=h_ratio,
        overlap_width_ratio=w_ratio,
        postprocess_class_agnostic=True,
        postprocess_type='NMM'
    )
    return preds

def load_labels(test_labels_path):
    with open(test_labels_path, 'r') as f:
        lines = f.readlines()

    # Convert text to numpy array
    label_data = np.array([list(map(float, line.split())) for line in lines])
    # Create a PyTorch tensor from the numpy array
    label_tensor = torch.from_numpy(label_data)
    return label_tensor

class SquarePad:
    """
    Pad all detections to 224 x 224 before encoding
    Rescales images larger than 224 while preserving aspect ratio
    """
    def __call__(self, image, img_size=224):
        c, h, w = image.size()
        
        # Scale image while preserving aspect ratio if larger than img_size
        if h > img_size or w > img_size:
            scale_factor = img_size / max(h, w)
            h, w = int(h * scale_factor), int(w * scale_factor)
            image = v2.functional.resize(image, (h, w))
        
        hp = (img_size - w) // 2
        vp = (img_size - h) // 2
        padding = (hp, img_size - w - hp, vp, img_size - h - vp) # left, right, top, bottom

        return v2.functional.pad(image, padding, fill=1)

