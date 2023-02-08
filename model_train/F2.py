from torchmetrics.metric import Metric
import torchvision
import torch
import numpy as np

from f_beta import f_beta


class F2(Metric):
    def __init__(
            self,
            compute_on_step=True,
            dist_sync_on_step=False,
            process_group=None,
            dist_sync_fn=None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("detection_boxes", default=[], dist_reduce_fx=None)
        self.add_state("detection_scores", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_boxes", default=[], dist_reduce_fx=None)

    def update(self, preds, target):
        for item in preds:
            self.detection_boxes.append(
                torchvision.ops.box_convert(item["boxes"], in_fmt="xywh", out_fmt="xyxy")
                if len(item["boxes"]) > 0
                else item["boxes"]
            )
            self.detection_scores.append(item["scores"])

        for item in target:
            self.groundtruth_boxes.append(
                torchvision.ops.box_convert(item["boxes"], in_fmt="xywh", out_fmt="xyxy")
                if len(item["boxes"]) > 0
                else item["boxes"]
            )

    def compute(self):
        tps, fps, fns = 0, 0, 0
        for gt_boxes, pred_boxes, pred_scores in zip(
                self.groundtruth_boxes, self.detection_boxes, self.detection_scores
        ):
            tp, fp, fn = self._compute_stat_scores(gt_boxes, pred_boxes, pred_scores)
            tps += tp
            fps += fp
            fns += fn

        return f_beta(tps, fps, fns, beta=2)

    def _compute_stat_scores(self, gt_boxes, pred_boxes, pred_scores):
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            tps, fps, fns = 0, 0, 0
            return tps, fps, fns

        elif len(gt_boxes) == 0:
            tps, fps, fns = 0, len(pred_boxes), 0
            return tps, fps, fns

        elif len(pred_boxes) == 0:
            tps, fps, fns = 0, 0, len(gt_boxes)
            return tps, fps, fns

        # sort by conf
        _, indices = torch.sort(pred_scores, descending=True)
        pred_boxes = pred_boxes[indices]

        tps, fps, fns = 0, 0, 0
        for iou_th in np.arange(0.3, 0.85, 0.05):
            tp, fp, fn = self._compute_stat_scores_at_iou_th(gt_boxes, pred_boxes, iou_th)
            tps += tp
            fps += fp
            fns += fn

        return tps, fps, fns

    def _compute_stat_scores_at_iou_th(self, gt_boxes, pred_boxes, iou_th):
        gt_boxes = gt_boxes.clone()
        pred_boxes = pred_boxes.clone()

        tp = 0
        fp = 0
        for k, pred_bbox in enumerate(pred_boxes):
            ious = torchvision.ops.box_iou(gt_boxes, pred_bbox[None, ...])

            max_iou = ious.max()
            if max_iou > iou_th:
                tp += 1

                # Delete max_iou box
                argmax_iou = ious.argmax()
                gt_boxes = torch.cat([gt_boxes[0:argmax_iou], gt_boxes[argmax_iou + 1:]])
            else:
                fp += 1
            if len(gt_boxes) == 0:
                fp += len(pred_boxes) - (k + 1)
                break

        fn = len(gt_boxes)

        return tp, fp, fn
