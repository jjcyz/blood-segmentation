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
PATCH_DEPTH = 4
BATCH_SIZE = 8
NUM_WORKERS = 4
CACHE_SIZE = 150

class DataPreprocessor:
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

    # kidney_dirs = ['kidney_1_dense', 'kidney_2', 'kidney_3_sparse']
    kidney_dirs = ['kidney_1_dense']

    for kidney_dir in kidney_dirs:
      img_paths, lbl_paths = self._process_kidney_dir(kidney_dir)
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

class MemoryEfficientSequence(tf.keras.utils.Sequence):
  def __init__(self, image_paths, label_paths, batch_size=BATCH_SIZE):
    super().__init__()
    self.image_paths = [p for p in image_paths if os.path.exists(p)]
    self.label_paths = [p for p in label_paths if os.path.exists(p)]
    self.batch_size = batch_size
    self.cache = {}
    self.cache_size = CACHE_SIZE
    self.valid_starts = self._find_valid_sequences()

    if len(self.valid_starts) == 0:
      raise ValueError("No valid sequences found in the dataset")

    print(f"Total images found {len(self.image_paths)}")
    print(f"Valid 3D sequence starts: {len(self.valid_starts)}")
    print(f"image_paths: {self.image_paths}")
    print(f"label_paths: {self.label_paths}")
    print(f"PATCH_DEPTH: {PATCH_DEPTH}")
    print(f"valid_starts: {self.valid_starts}")
    self.on_epoch_end()

  def _find_valid_sequences(self):
    valid_starts = []
    for i in range(len(self.image_paths) - PATCH_DEPTH + 1):
      sequence_valid = True
      for j in range(PATCH_DEPTH):
        idx = i + j
        if not (os.path.exists(self.image_paths[idx]) and
                os.path.exists(self.label_paths[idx])):
          sequence_valid = False
          break
      if sequence_valid:
        valid_starts.append(i)
    return np.array(valid_starts)

  def __len__(self):
    return int(np.ceil(len(self.valid_starts) / self.batch_size))

  def __get_cached_image(self, path):
    if path in self.cache:
      return self.cache[path]

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
      return np.zeros((PATCH_HEIGHT, PATCH_WIDTH), dtype=np.float32)

    img = cv2.resize(img, (PATCH_WIDTH, PATCH_HEIGHT))
    img = img.astype(np.float32) / 255.0
    if len(self.cache) >= self.cache_size:
      self.cache.pop(list(self.cache.keys())[0])
    self.cache[path] = img
    return img

  def __getitem__(self, idx):
    batch_starts = self.valid_starts[idx * self.batch_size:(idx + 1) * self.batch_size]
    batch_x = np.zeros((len(batch_starts), PATCH_DEPTH, PATCH_HEIGHT, PATCH_WIDTH, 1), dtype=np.float32)
    batch_y = np.zeros((len(batch_starts), PATCH_DEPTH, PATCH_HEIGHT, PATCH_WIDTH, 1), dtype=np.float32)

    for i, start_idx in enumerate(batch_starts):
      volume_x = np.zeros((PATCH_DEPTH, PATCH_HEIGHT, PATCH_WIDTH))
      volume_y = np.zeros((PATCH_DEPTH, PATCH_HEIGHT, PATCH_WIDTH))

      for j in range(PATCH_DEPTH):
        idx = start_idx + j
        volume_x[j] = self.__get_cached_image(self.image_paths[idx])
        volume_y[j] = self.__get_cached_image(self.label_paths[idx])

      batch_x[i] = volume_x[..., np.newaxis]
      batch_y[i] = volume_y[..., np.newaxis]

    return batch_x, batch_y

  def on_epoch_end(self):
    np.random.shuffle(self.valid_starts)
    self.cache.clear()

  def __del__(self):
    self.cache.clear()
