import os
import gc
import cv2
import numpy as np
import tensorflow as tf
from glob import glob
from concurrent.futures import ThreadPoolExecutor

# Constants
PATCH_WIDTH = 64
PATCH_HEIGHT = 64
PATCH_DEPTH = 16
BATCH_SIZE = 8
NUM_WORKERS = 4
CACHE_SIZE = 150

class DataProcessor:
  def __init__(self, input_dir):
    self.input_dir = input_dir
    self.thread_pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)

  def get_file_paths(self, data_type='train'):
    print(f"Searching for {data_type} data in: {self.input_dir}")

    if data_type == 'train':
      return self._get_train_file_paths()
    else:
      return self._get_test_paths()

  def _get_train_file_paths(self):
    image_paths = []
    label_paths = []

    kidney_dirs = ['kidney_1_dense', 'kidney_2', 'kidney_3_sparse']

    for kidney_dir in kidney_dirs:
      img_paths, lbl_paths = self._get_kidney_paths(kidney_dir)
      if img_paths and lbl_paths:
        image_paths.extend(img_paths)
        label_paths.extend(lbl_paths)

    return image_paths, label_paths


