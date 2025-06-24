# config.py

import os

class Config:
    # ===== General Hyperparameters =====
    IMAGE_SIZE = 224  # Not directly used for resizing in detection; safe default
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # Normalization values (ImageNet defaults)
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    # ===== Training =====
    EPOCHS = 20

    # ===== Paths =====
    BASE_PATH = "/media/jag/volD/BID_DATA/obd/final_OBD"

    TRAIN_DATA_PATH        = os.path.join(BASE_PATH, "train")
    VAL_DATA_PATH          = os.path.join(BASE_PATH, "val")
    TRAIN_ANNOTATION_PATH  = os.path.join(BASE_PATH, "annotations", "lvis_single_object_train.json")
    VAL_ANNOTATION_PATH    = os.path.join(BASE_PATH, "annotations", "lvis_single_object_val.json")
