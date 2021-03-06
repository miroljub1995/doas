# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os

import cv2
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from keras.utils import multi_gpu_model

from api.yolo.yolo3.model import yolo_eval, yolo_body, tiny_yolo_body

class YOLO(object):
    _defaults = {
        "model_path": 'api/yolo/model_data/trained_weights_final.h5',
        "anchors_path": 'api/yolo/model_data/tiny_yolo_anchors.txt',
        "classes_path": 'api/yolo/model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416), # (h, w)
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect(self, image):
        shape = (self.model_image_size[0], self.model_image_size[1], 3)
        assert shape == image.shape, "The shape of image should be: {}, but got: {}".format(shape, image.shape)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_data = np.array(image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [shape[0], shape[1]],
                K.learning_phase(): 0
            })

        for box in out_boxes:
            box[0], box[1] = box[1], box[0]
            box[2], box[3] = box[3], box[2]

        return (out_boxes, out_scores, out_classes)

    def draw(self, image, detection):
        h, w, _channels = image.shape
        out_boxes, out_scores, out_classes = detection

        cv_font = cv2.FONT_HERSHEY_SIMPLEX
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            cv_label_w, cv_label_h = cv2.getTextSize(label, cv_font, 0.5, 1)[0]

            left, top, right, bottom  = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(h, np.floor(bottom + 0.5).astype('int32'))
            right = min(w, np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - cv_label_h >= 0:
                text_origin = np.array([left, top - cv_label_h])
            else:
                text_origin = np.array([left, top + 1])

            cv2.rectangle(image, (left + i, top + i), (right - i, bottom - i), self.colors[c], 1)
            cv2.rectangle(image, tuple(text_origin), tuple(text_origin + [cv_label_w, cv_label_h]), self.colors[c], -1)
            cv2.putText(image, label, tuple(text_origin + [0, cv_label_h]), cv_font, 0.5, (0, 0, 0), 1)

def create():
    return YOLO()

def lines_and_obstacles(objs, labels):
    lines = []
    obstacles = []
    for obj in zip(objs, labels):
        if obj[1] == 0:
            lines.append(obj[0])
        else:
            obstacles.append(obj[0])
    return [lines, obstacles]

def calculate_goal(lines):
    left = []
    right = []
    if lines[0][0] < lines[1][0] and lines[0][2] < lines[1][3]:
        left = lines[0]
        right = lines[1]
    else:
        right = lines[0]
        left = lines[1]

    l_y = (left[1] + left[3]) / 2
    r_y = (right[1] + right[3]) / 2
    middle = (int((left[2] + right[0]) / 2), int((l_y + r_y) / 2))
    return middle