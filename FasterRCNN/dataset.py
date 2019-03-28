# -*- coding: utf-8 -*-
# File: coco.py

import numpy as np
import os
import tqdm
import json
from evaluate import *
from tensorpack.utils import logger
from tensorpack.utils.timer import timed_operation
import pickle

from config import config as cfg

import pandas as pd

__all__ = ['COCODetection', 'DetectionDataset']


class COCODetection:
    # handle the weird (but standard) split of train and val
    _INSTANCE_TO_BASEDIR = {
        'valminusminival2014': 'val2014',
        'minival2014': 'val2014',
    }

    COCO_id_to_category_id = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72, 82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}  # noqa
    """
    Mapping from the incontinuous COCO category id to an id in [1, #category]
    For your own dataset, this should usually be an identity mapping.
    """

    # 80 names for COCO
    class_names = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]  # noqa

    def __init__(self, basedir, name):
        basedir = os.path.expanduser(basedir)
        self.name = name
        self._imgdir = os.path.realpath(os.path.join(
            basedir, self._INSTANCE_TO_BASEDIR.get(name, name)))
        assert os.path.isdir(self._imgdir), self._imgdir
        annotation_file = os.path.join(
            basedir, 'annotations/instances_{}.json'.format(name))
        assert os.path.isfile(annotation_file), annotation_file

        from pycocotools.coco import COCO
        self.coco = COCO(annotation_file)
        logger.info("Instances loaded from {}.".format(annotation_file))

    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    def print_coco_metrics(self, json_file):
        """
        Args:
            json_file (str): path to the results json file in coco format
        Returns:
            dict: the evaluation metrics
        """
        from pycocotools.cocoeval import COCOeval
        ret = {}
        cocoDt = self.coco.loadRes(json_file)
        cocoEval = COCOeval(self.coco, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        fields = ['IoU=0.5:0.95', 'IoU=0.5', 'IoU=0.75', 'small', 'medium', 'large']
        for k in range(6):
            ret['mAP(bbox)/' + fields[k]] = cocoEval.stats[k]

        json_obj = json.load(open(json_file))
        if len(json_obj) > 0 and 'segmentation' in json_obj[0]:
            cocoEval = COCOeval(self.coco, cocoDt, 'segm')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            for k in range(6):
                ret['mAP(segm)/' + fields[k]] = cocoEval.stats[k]
        return ret

    def load(self, add_gt=True, add_mask=False):
        """
        Args:
            add_gt: whether to add ground truth bounding box annotations to the dicts
            add_mask: whether to also add ground truth mask

        Returns:
            a list of dict, each has keys including:
                'image_id', 'file_name',
                and (if add_gt is True) 'boxes', 'class', 'is_crowd', and optionally
                'segmentation'.
        """
        if add_mask:
            assert add_gt
        with timed_operation('Load Groundtruth Boxes for {}'.format(self.name)):
            img_ids = self.coco.getImgIds()
            img_ids.sort()
            # list of dict, each has keys: height,width,id,file_name
            imgs = self.coco.loadImgs(img_ids)

            for img in tqdm.tqdm(imgs):
                img['image_id'] = img.pop('id')
                self._use_absolute_file_name(img)
                if add_gt:
                    self._add_detection_gt(img, add_mask)
            return imgs

    def _use_absolute_file_name(self, img):
        """
        Change relative filename to abosolute file name.
        """
        img['file_name'] = os.path.join(
            self._imgdir, img['file_name'])
        assert os.path.isfile(img['file_name']), img['file_name']

    def _add_detection_gt(self, img, add_mask):
        """
        Add 'boxes', 'class', 'is_crowd' of this image to the dict, used by detection.
        If add_mask is True, also add 'segmentation' in coco poly format.
        """
        # ann_ids = self.coco.getAnnIds(imgIds=img['image_id'])
        # objs = self.coco.loadAnns(ann_ids)
        objs = self.coco.imgToAnns[img['image_id']]  # equivalent but faster than the above two lines

        # clean-up boxes
        valid_objs = []
        width = img.pop('width')
        height = img.pop('height')
        for objid, obj in enumerate(objs):
            if obj.get('ignore', 0) == 1:
                continue
            x1, y1, w, h = obj['bbox']
            # bbox is originally in float
            # x1/y1 means upper-left corner and w/h means true w/h. This can be verified by segmentation pixels.
            # But we do make an assumption here that (0.0, 0.0) is upper-left corner of the first pixel

            x1 = np.clip(float(x1), 0, width)
            y1 = np.clip(float(y1), 0, height)
            w = np.clip(float(x1 + w), 0, width) - x1
            h = np.clip(float(y1 + h), 0, height) - y1
            # Require non-zero seg area and more than 1x1 box size
            if obj['area'] > 1 and w > 0 and h > 0 and w * h >= 4:
                obj['bbox'] = [x1, y1, x1 + w, y1 + h]
                valid_objs.append(obj)

                if add_mask:
                    segs = obj['segmentation']
                    if not isinstance(segs, list):
                        assert obj['iscrowd'] == 1
                        obj['segmentation'] = None
                    else:
                        valid_segs = [np.asarray(p).reshape(-1, 2).astype('float32') for p in segs if len(p) >= 6]
                        if len(valid_segs) == 0:
                            logger.error("Object {} in image {} has no valid polygons!".format(objid, img['file_name']))
                        elif len(valid_segs) < len(segs):
                            logger.warn("Object {} in image {} has invalid polygons!".format(objid, img['file_name']))

                        obj['segmentation'] = valid_segs

        # all geometrically-valid boxes are returned
        boxes = np.asarray([obj['bbox'] for obj in valid_objs], dtype='float32')  # (n, 4)
        cls = np.asarray([
            self.COCO_id_to_category_id[obj['category_id']]
            for obj in valid_objs], dtype='int32')  # (n,)
        is_crowd = np.asarray([obj['iscrowd'] for obj in valid_objs], dtype='int8')

        # add the keys
        img['boxes'] = boxes        # nx4
        img['class'] = cls          # n, always >0
        img['is_crowd'] = is_crowd  # n,
        if add_mask:
            # also required to be float32
            img['segmentation'] = [
                obj['segmentation'] for obj in valid_objs]

    @staticmethod
    def load_many(basedir, names, add_gt=True, add_mask=False):
        """
        Load and merges several instance files together.

        Returns the same format as :meth:`COCODetection.load`.
        """
        if not isinstance(names, (list, tuple)):
            names = [names]
        ret = []
        for n in names:
            coco = COCODetection(basedir, n)
            ret.extend(coco.load(add_gt, add_mask=add_mask))
        return ret


class DetectionDataset:
    """
    A singleton to load datasets, evaluate results, and provide metadata.

    To use your own dataset that's not in COCO format, rewrite all methods of this class.
    """

    def __init__(self):
        """
        This function is responsible for setting the dataset-specific
        attributes in both cfg and self.
        """
        self.read_file()

    def read_file(self):
        self.df = pd.read_hdf(cfg.DATA.DATAFRAME)
        self.class_names = cfg.DATA.CLASS_NAMES = ["BG"] + list(self.df["category"].unique())
        self.num_category = cfg.DATA.NUM_CATEGORY = len(self.class_names) - 1
        self.num_classes = 1 + self.num_category
        temp = np.array([self.class_names, [k for k in range(self.num_classes)]])
        temp = np.transpose(temp)
        temp = pd.DataFrame(temp, columns=["category", "class"])
        self.df = self.df.merge(temp, on="category")

    def load_training_roidbs(self, names):
        """
        Args:
            names (list[str]): name of the training datasets, e.g.  ['train2014', 'valminusminival2014']

        Returns:
            roidbs (list[dict]):

        Produce "roidbs" as a list of dict, each dict corresponds to one image with k>=0 instances.
        and the following keys are expected for training:

        file_name: str, full path to the image
        boxes: numpy array of kx4 floats, each row is [x1, y1, x2, y2]
        class: numpy array of k integers, in the range of [1, #categories], NOT [0, #categories)
        is_crowd: k booleans. Use k False if you don't know what it means.
        segmentation: k lists of numpy arrays (one for each instance).
            Each list of numpy arrays corresponds to the mask for one instance.
            Each numpy array in the list is a polygon of shape Nx2,
            because one mask can be represented by N polygons.

            If your segmentation annotations are originally masks rather than polygons,
            either convert it, or the augmentation will need to be changed or skipped accordingly.

            Include this field only if training Mask R-CNN.
        """
        roidbs = []

        def build_roidbs(df):
            def read_row(row):
                boxes.append([row["xmin"], row["ymin"], row["xmax"], row["ymax"]])
                classes.append(row["class"])
                is_crowd.append(False)
            temp = {}
            temp["file_name"] = df.name
            boxes = []
            classes = []
            is_crowd = []

            df.apply(read_row, axis=1)
            temp["boxes"] = np.array(boxes).astype(np.float32)
            temp["class"] = np.array(classes).astype(np.int)
            temp["is_crowd"] = np.array(is_crowd)

            roidbs.append(temp)

        self.df[self.df["type"].isin(names)].groupby("image_path").apply(build_roidbs)
        # print(roidbs)
        return roidbs

    def load_inference_roidbs(self, name):
        """
        Args:
            name (str): name of one inference dataset, e.g. 'minival2014'

        Returns:
            roidbs (list[dict]):

            Each dict corresponds to one image to run inference on. The
            following keys in the dict are expected:

            file_name (str): full path to the image
            image_id (str): an id for the image. The inference results will be stored with this id.
        """
        inference = []

        def build_inference(row):
            temp = {}
            temp["file_name"] = row["image_path"]
            temp["image_id"] = row.name
            inference.append(temp)

        self.df[self.df["type"] == name][["image_path"]].drop_duplicates().apply(build_inference, axis=1)
        return inference

    def eval_or_save_inference_results(self, results, dataset, output=None):
        """
        Args:
            results (list[dict]): the inference results as dicts.
                Each dict corresponds to one __instance__. It contains the following keys:

                image_id (str): the id that matches `load_inference_roidbs`.
                category_id (int): the category prediction, in range [1, #category]
                bbox (list[float]): x1, y1, x2, y2
                score (float):
                segmentation: the segmentation mask in COCO's rle format.

            dataset (str): the name of the dataset to evaluate.
            output (str): the output file to optionally save the results to.

        Returns:
            dict: the evaluation results.
        """
        ids = self.df[self.df["type"] == dataset][["image_path"]].drop_duplicates().reset_index()
        ids.columns = ["ids", "image_path"]

        allBB = BoundingBoxes.BoundingBoxes()
        # allClasses = self.class_names

        def buildBB(row, truth=False):
            if not truth:
                bb = BoundingBox.BoundingBox(
                    ids[ids["ids"] == row["image_id"]].iloc[0]["image_path"],
                    self.class_names[int(row["category_id"])],
                    row["bbox"][0],
                    row["bbox"][1],
                    row["bbox"][2],
                    row["bbox"][3],
                    utils.CoordinatesType.Absolute,
                    (0, 0),
                    utils.BBType.Detected,
                    row["score"],
                    format=utils.BBFormat.XYX2Y2
                )
            else:
                bb = BoundingBox.BoundingBox(
                    row["image_path"],
                    self.class_names[int(row["class"])],
                    row["xmin"],
                    row["ymin"],
                    row["xmax"],
                    row["ymax"],
                    utils.CoordinatesType.Absolute,
                    (0, 0),
                    utils.BBType.GroundTruth,
                    format=utils.BBFormat.XYX2Y2
                )
            allBB.addBoundingBox(bb)

        for r in results:
            buildBB(r)
        self.df.apply(lambda r: buildBB(r, True), axis=1)

        evaluator = Evaluator.Evaluator()
        evaluation = evaluator.GetPascalVOCMetrics(allBB)
        result = {}
        total_AP = 0
        count_classes = 0
        for e in evaluation:
            c = e["class"]
            if c != "others":
                result["precision/" + c] = e["precision"] if type(e["precision"]) is not np.ndarray else 0
                result["recall/" + c] = e["recall"] if type(e["recall"]) is not np.ndarray else 0
                result["AP/" + c] = e["AP"] if type(e["AP"]) is not np.ndarray else 0
                total_AP += e["AP"] if type(e["AP"]) is not np.ndarray else 0
                count_classes += 1
        result["mAP"] = total_AP / count_classes
        try:
            if output is not None:
                # with open(output + ".raw.json", 'w') as f:
                #    json.dump(results, f)
                with open(output, 'w') as f:
                    json.dump(result, f)
        except:
            pass
        for k in range(len(results)):
            results[k]["image_id"] = ids[ids["ids"] == results[k]["image_id"]].iloc[0]["image_path"]
            results[k]["category"] = self.class_names[int(results[k]["category_id"])]
        with open("out.pickle", "wb") as f:
            pickle.dump(results, f)
        return result

    # code for singleton:
    _instance = None

    def __new__(cls):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls)
        return cls._instance

    def get_iou(bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner

        Returns
        -------
        float
            in [0, 1]
        """
        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou


if __name__ == '__main__':
    cfg.DATA.BASEDIR = '~/data/coco'
    c = COCODetection(cfg.DATA.BASEDIR, 'train2014')
    roidb = c.load(add_gt=True, add_mask=True)
    print("#Images:", len(roidb))
