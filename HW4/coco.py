"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
from imgaug.augmenters import Affine, Sequential, Fliplr, GaussianBlur, Multiply, AdditiveGaussianNoise, MultiplyHue, AddToHueAndSaturation, Flipud, GammaContrast, Sharpen, OneOf, Sometimes, CoarseDropout

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize
import imgaug as ia
import imgaug.augmenters as iaa
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "pascal_train.json")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = ""

############################################################
#  Configurations
############################################################
import random

class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "HW4"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    #IMAGES_PER_GPU = 2


    # Into crop
    IMAGES_PER_GPU = 4
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    IMAGE_MIN_SCALE = 1


    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 20  # COCO has 80 classes


############################################################
#  Dataset
############################################################
import numpy as np
from itertools import groupby
from pycocotools import mask as maskutil
import json
def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    compressed_rle = maskutil.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    compressed_rle['counts'] = str(compressed_rle['counts'], encoding='utf-8')
    return compressed_rle

class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, class_ids=None,
                  class_map=None, return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        coco = COCO("{}/annotations/instances_{}.json".format(dataset_dir, subset))

        image_dir = "{}/{}".format(dataset_dir, subset)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Testing data json is empty
        if not image_ids:
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################

import scipy
import warnings
def my_resize(mask, scale, crop=None):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    (t, b), (l, r), (_,_) = crop
    w = mask.shape[0]
    h = mask.shape[1]
    mask = mask[t:h-b, l:w-r]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scale = 1/scale
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    return mask

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CocoConfig()
    else:
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            IMAGE_RESIZE_MODE = "square"
            IMAGE_MAX_DIM = 256

            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.86
            DETECTION_NMS_THRESHOLD = 0.3
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = CocoDataset()
        dataset_train.load_coco(args.dataset, "train")
        dataset_train.prepare()

        # Validation dataset

        # 2017 valset
        dataset_val = CocoDataset()
        dataset_val.load_coco(args.dataset, "val2007")
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time, IOU50=0.029(VALIDATION SET)
        # augmentation = imgaug.augmenters.Fliplr(0.5)

        # Flip and rotate
        augmentation = Sequential([Fliplr(0.5), GaussianBlur((0, 1.5)), CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05)), Affine(scale=(0.3, 2), rotate=(-20, 20)), Multiply((0.5,1.5)), AddToHueAndSaturation((-20,20))])
        #augmentation = Sequential([Flipud(0.2), Fliplr(0.5), Affine(scale=(0.2, 2), rotate=(-60,60)), GaussianBlur((0, 2)), AddToHueAndSaturation((-30,30)), GammaContrast((0.5, 1.5)), AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)])
        #augmentation = Sequential([Fliplr(0.5), Affine(scale=(0.3, 3), rotate=(-60,60)), GaussianBlur((0, 3)), AddToHueAndSaturation((-30,30)), GammaContrast((0.5, 1.5)), AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5) ])

        '''
            iaa.Sequential([
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.1),  # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width

            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                       [
                           sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                           # convert images into their superpixel representation
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                               iaa.AverageBlur(k=(2, 7)),
                               # blur image using local means with kernel sizes between 2 and 7
                               iaa.MedianBlur(k=(3, 11)),
                               # blur image using local medians with kernel sizes between 2 and 7
                           ]),
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                           iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                           # search either for all edges or for directed edges,
                           # blend the result with the original image using a blobby mask
                           iaa.SimplexNoiseAlpha(iaa.OneOf([
                               iaa.EdgeDetect(alpha=(0.5, 1.0)),
                               iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                           ])),
                           iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                           # add gaussian noise to images
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                               iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                           ]),
                           iaa.Invert(0.05, per_channel=True),  # invert color channels
                           iaa.Add((-10, 10), per_channel=0.5),
                           # change brightness of images (by -10 to 10 of original value)
                           iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                           # either change the brightness of the whole image (sometimes
                           # per channel) or change the brightness of subareas
                           iaa.OneOf([
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               iaa.FrequencyNoiseAlpha(
                                   exponent=(-4, 0),
                                   first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                   second=iaa.LinearContrast((0.5, 2.0))
                               )
                           ]),
                           iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                           iaa.Grayscale(alpha=(0.0, 1.0)),
                       ],
                       random_order=True
                       )
        ],
            random_order=True)
        '''

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1

        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        #print("Fine tune Resnet stage 4 and up")
        #model.train(dataset_train, dataset_val,
        #           learning_rate=config.LEARNING_RATE/10,
        #           epochs=120,
        #           layers='4+',
        #           augmentation=augmentation)
        '''
        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all',
                    augmentation=augmentation)
        '''

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = CocoDataset()
        val_type = "val2007"
        coco = dataset_val.load_coco(args.dataset, val_type, return_coco=True)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
    elif args.command == 'test':
        dataset = CocoDataset()
        dataset.load_coco(args.dataset, 'test')
        dataset.prepare()
        while 1:
            image_id = random.choice(dataset.image_ids)
            image = dataset.load_image(image_id)
            original_shape = image.shape
            image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=config.IMAGE_MIN_DIM,
                min_scale=config.IMAGE_MIN_SCALE,
                max_dim=config.IMAGE_MAX_DIM,
                mode=config.IMAGE_RESIZE_MODE)

            info = dataset.image_info[image_id]
            print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                                   dataset.image_reference(image_id)))
            # Run object detection
            results = model.detect([image], verbose=1)
            # Display results

            #print(len(results))
            #print(results[0].keys())
            #print(results[0]['rois'])
            #print(results[0]['masks'].shape)
            #input()

            r = results[0]
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                        dataset.class_names, r['scores'],
                                        title="Predictions")
    elif args.command == 'out':
        coco_dt = []
        cocoGt = COCO("./train_images/annotations/instances_test.json")
        import cv2
        import skimage.io
        for imgid in cocoGt.imgs:
            image = skimage.io.imread("./train_images/test/" + cocoGt.loadImgs(ids=imgid)[0]['file_name'])
            original_shape = image.shape
            old_image = image
            image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=config.IMAGE_MIN_DIM,
                min_scale=config.IMAGE_MIN_SCALE,
                max_dim=config.IMAGE_MAX_DIM,
                mode=config.IMAGE_RESIZE_MODE)

            #print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
            #                                       dataset.image_reference(image_id)))
            # Run object detection
            results = model.detect([image], verbose=1)
            print(results[0]['masks'].shape)
            # Display results
            print(scale)

            # Back scale the mask
            mask = my_resize(results[0]['masks'], scale, padding)
            # Make fit to image
            # Crop and pad
            if mask.shape[0]>original_shape[0]:
                mask = mask[0:original_shape[0]]
            elif mask.shape[0]<original_shape[0]:
                mask = np.pad(mask, [(0, original_shape[0] - mask.shape[0]), (0, 0), (0, 0)], mode='constant',
                              constant_values=0)
            if mask.shape[1]>original_shape[1]:
                mask = mask[:, 0:original_shape[0]]
            elif mask.shape[1]<original_shape[1]:
                mask = np.pad(mask, [(0, 0), (0, original_shape[1] - mask.shape[1]), (0, 0)], mode='constant',
                              constant_values=0)
            #input()
            for i in range(mask.shape[2]):  # Loop all instances
                # save information of the instance in a dictionary then append on coco_dt list
                pred = {}
                pred['image_id'] = imgid  # this imgid must be same as the key of test.json
                pred['category_id'] = int(results[0]['class_ids'][i])
                pred['segmentation'] = binary_mask_to_rle(
                    mask[:, :, i])  # save binary mask to RLE, e.g. 512x512 -> rle
                pred['score'] = float(results[0]['scores'][i])
                coco_dt.append(pred)
            # r = results[0]
            # visualize.display_instances(old_image, r['rois'], mask, r['class_ids'],
            #                            dataset.class_names, r['scores'],
            #                            title="Predictions")

        with open("submission.json", "w") as f:
            json.dump(coco_dt, f)

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
