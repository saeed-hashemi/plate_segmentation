from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import time
import numpy as np

def model_initialize(model_weights_address):
    # Inference should use the config with parameters that are used in training
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("plate_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 6000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon).
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 4  # 17 is the number of keypoints in COCO.
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = model_weights_address  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.90   # set a custom testing threshold
    cfg.MODEL.DEVICE='cpu'
    return cfg

class inference:
  def __init__(self, model_weights_address):
    self.w = model_weights_address
    
  def load(self):
    cfg=model_initialize(model_weights_address=self.w)
    self.predictor = DefaultPredictor(cfg)
  def predict(self, image):
    start_d=time.time()
    outputs = self.predictor(image)
    print(f"time (detection): {time.time()-start_d}")
    if outputs["instances"].pred_masks.shape[0] != 0:
      mask = outputs["instances"].pred_masks[0,:,:].cpu().numpy().astype(np.uint8)*255
      return mask
    else:
      return None