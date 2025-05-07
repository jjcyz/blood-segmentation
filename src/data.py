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

  def _process_kidney_dir(self, kidney_dir):
			base_path = os.path.join(self.input_dir, 'train', kidney_dir)
			print(f"Checking directory: {base_path}")

			if not os.path.exists(base_path):
					return [], []

			img_dir = os.path.join(base_path, 'images')
			label_dir = os.path.join(base_path, 'labels')

			img_paths = sorted(glob(os.path.join(img_dir, '*.tif')))
			label_paths = sorted(glob(os.path.join(label_dir, '*.tif')))

			print(f"Found {len(img_paths)} images and {len(label_paths)} labels in {kidney_dir}")

			return img_paths, label_paths

  def _get_test_paths(self):
    test_path = os.path.join(self.input_dir, 'test')
    return sorted(glob(os.path.join(test_path, '*/images/*.tif')))

  def __del__(self):
    self.thread_pool.shutdown()
