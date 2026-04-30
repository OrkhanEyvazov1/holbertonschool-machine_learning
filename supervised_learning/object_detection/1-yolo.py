#!/usr/bin/env python3
"""Yolo v3 object detection model."""
import numpy as np
import tensorflow.keras as K


class Yolo(K.Model):
    """Yolo v3 object detection model."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize the Yolo model."""
        super().__init__()
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            class_names = [line.strip() for line in f]
        object.__setattr__(self, 'class_names', class_names)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process Darknet model outputs for a single image."""
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size
        model_height = self.model.input.shape[1]
        model_width = self.model.input.shape[2]

        for output, anchor_boxes in zip(outputs, self.anchors):
            grid_height, grid_width, num_anchors, _ = output.shape

            box = np.zeros((grid_height, grid_width, num_anchors, 4))

            x_indices = np.arange(grid_width)
            y_indices = np.arange(grid_height).reshape(-1, 1)

            x_center = (
                (1 / (1 + np.exp(-output[..., 0]))) + x_indices
            ) * image_width / grid_width
            y_center = (
                (1 / (1 + np.exp(-output[..., 1]))) + y_indices
            ) * image_height / grid_height
            box_width = (
                anchor_boxes[..., 0]
                * np.exp(output[..., 2])
                * image_width
                / model_width
            )
            box_height = (
                anchor_boxes[..., 1]
                * np.exp(output[..., 3])
                * image_height
                / model_height
            )

            box[..., 0] = x_center - box_width / 2
            box[..., 1] = y_center - box_height / 2
            box[..., 2] = x_center + box_width / 2
            box[..., 3] = y_center + box_height / 2

            boxes.append(box)
            box_confidences.append(1 / (1 + np.exp(-output[..., 4:5])))
            box_class_probs.append(1 / (1 + np.exp(-output[..., 5:])))

        return boxes, box_confidences, box_class_probs
