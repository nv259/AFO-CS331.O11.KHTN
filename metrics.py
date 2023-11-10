import torch
import numpy as np

from collections import Counter


def IoU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


def mean_average_precision(
    pred_boxes, gt_boxes, iou_threshold=0.5, box_format="corners", classes=["small", "large"], writer=None
):
    # pred_boxes (list): [[image_id, class_pred, prob_score, x1, y1, x2, y2], ...]
    average_precisions = []
    epsilon = 1e-6
    num_classes = len(classes)
    
    for c in range(num_classes):
        detections = []
        ground_truths = []
        
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
        
        for gt_box in gt_boxes:
            if gt_box[1] == c:
                ground_truths.append(gt_box)
        
        # img 0 has 3 bboxes
        # img 1 has 5 bboxes
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        
        # amount_bboxes = {0: torch.tensor([0, 0, 0]), 1: torch.tensor([0, 0, 0, 0, 0])}
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]
            
            num_gts = len(ground_truth_img)
            best_iou = 0
            
            for idx, gt in enumerate(ground_truth_img):
                iou = IoU(
                    detection[3:],
                    gt[3:],
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            
            else:
                FP[detection_idx] = 1
            
        # [1, 1, 0, 1, 0] - > [1, 2, 2, 3, 3]
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))
        
        if writer is not None:
            for idx, precision in enumerate(precisions):
                writer.add_scalar('precision', precision.item(), idx)
            
            for idx, recall in enumerate(recalls):
                writer.add_scalar('recall', recall.item(), idx)
        
        print(f"AP@{iou_threshold} - {classes[c]}: {average_precisions[c]}")

    mAP = sum(average_precisions) / len(average_precisions)
    print(f"mAP@{iou_threshold}: {mAP}")
    
    return mAP


def mAP05_095(
    pred_boxes, gt_boxes, box_format="corners", classes=["small", "large"], writer=None
):
    mAPs = []
    
    for iou_threshold in range(0.5, 0.95, 0.05):
        mAPs.append(mean_average_precision(pred_boxes, gt_boxes, box_format=box_format, classes=classes, writer=writer))
    
    final_mAP = sum(mAPs) / len(mAPs)
    print(f"mAP@0.5:0.95 {final_mAP}")
    
    return final_mAP


def my_evaluate(model, loader, )