import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from PIL import Image

setup_logger()


def setup_config_faster_rcnn(
    train_set="plotqa_train",
    path_pretrained_weights="/home/charts-description/fastercnn_pretrained_weights/model_final_280758.pkl"
):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (train_set,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = path_pretrained_weights
    # learning rate
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = 20000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    # number of classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13
    # color representation
    cfg.INPUT.FORMAT = "RGB"
    # Size of the smallest side of the image during training
    cfg.INPUT.MIN_SIZE_TRAIN = (700,)
    # Maximum size of the side of the image during training
    cfg.INPUT.MAX_SIZE_TRAIN = 1100
    # Size of the smallest side of the image during testing
    cfg.INPUT.MIN_SIZE_TEST = 700
    # Maximum size of the side of the image during testing
    cfg.INPUT.MAX_SIZE_TEST = 1100
    return cfg


def setup_config_retinanet(
    train_set="plotqa_train",
    path_pretrained_weights="/home/charts-description/retinanet_pretrained_weights/model_final_5bd44e.pkl"
):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (train_set,)
    cfg.DATASETS.TEST = ()
    cfg.MODEL.WEIGHTS = path_pretrained_weights
    cfg.SOLVER.MAX_ITER = 20000
    cfg.SOLVER.IMS_PER_BATCH = 12
    # number of classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13
    cfg.MODEL.RETINANET.NUM_CLASSES = 13
    cfg.INPUT.FORMAT = "RGB"
    # Size of the smallest side of the image during training
    cfg.INPUT.MIN_SIZE_TRAIN = (700,)
    # Maximum size of the side of the image during training
    cfg.INPUT.MAX_SIZE_TRAIN = 1100
    # Size of the smallest side of the image during testing
    cfg.INPUT.MIN_SIZE_TEST = 700
    # Maximum size of the side of the image during testing
    cfg.INPUT.MAX_SIZE_TEST = 1100
    return cfg


def evaluate_model(
    model="faster-rcnn",
    path_model_weights="/home/charts-description/model_final.pth", 
    test_annotations="/home/charts-description/data/coco_annotations/coco_annotations_plotqa_test.json",
    test_images="/home/charts-description/data/plotqa/test1/png"
):
    register_coco_instances("test", {}, test_annotations, test_images)

    if model == "faster-rcnn":
        cfg = setup_config_faster_rcnn()
    else:
        cfg = setup_config_retinanet()

    cfg.MODEL.WEIGHTS = path_model_weights
    cfg.DATASETS.TEST = ("test", )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("test", cfg, False, output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "test")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))


def train_model(
    model="faster-rcnn",
    train_annotations="/home/charts-description/data/coco_annotations/coco_annotations_plotqa_train_sample_05.json",
    train_images="/home/charts-description/data/plotqa/train1/png"
):
    if model == "faster-rcnn":
        cfg = setup_config_faster_rcnn()
    else:
        cfg = setup_config_retinanet()
    # train
    register_coco_instances("plotqa_train", {}, train_annotations, train_images)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def plot_prediction(
    model_type,
    path_model_weights,
    img_path,
    dataset="plotqa",
    categories=None
):
    if categories is None:
        categories = {
            0: 'bar',
            1: 'dot line',
            2: 'legend', 
            3: 'line',
            4: 'title', 
            5: 'x label', 
            6: 'x tick label', 
            7: 'y label',
            8: 'y tick label',
            9: 'x axis',
            10: 'y axis',
            11: 'le',
            12: 'led'
        }

    if model_type == "faster-rcnn":
        cfg = setup_config_faster_rcnn()
    else:
        cfg = setup_config_retinanet()

    cfg.MODEL.WEIGHTS = path_model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    def add_rectangular(element, text):
        rect = patches.Rectangle(
            (element[0], element[1]),
            element[2],
            element[3],
            linewidth=1, edgecolor='r', facecolor='none'
            )
        ax.add_patch(rect)
        ax.annotate(text, (element[0], element[1]), color="gray")
    im = cv2.imread(img_path)
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    prediction = outputs['instances'].pred_classes
    boxes = outputs['instances'].pred_boxes
    img = Image.open(img_path)
    im = np.array(img, dtype=np.uint8)
    # create figure and axes
    fig, ax = plt.subplots(1, figsize=(15, 10))
    plt.axis('off')
    # display the image
    ax.imshow(im)
    for id, box in enumerate(boxes):
        box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
        add_rectangular(box, categories[int(prediction[id])])
    fig_name = img_path.split("/")[-1].split(".")[-2] + dataset +"_pred.png"
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
    plt.show()
