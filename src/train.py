import os
import gc
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision
from datetime import datetime


from .model import create_memory_efficient_unet
from .data import DataProcessor, load_and_preprocess_data
from .metrics import MetricsCollection
from .utils import PerformanceMonitor, MemoryCleanupCallback

mixed_precision.set_global_policy('mixed_float16')

# Constants
BATCH_SIZE = 8
VALIDATION_SPLIT = 0.2
WORKING_DIR = '/kaggle/working'
INPUT_DIR = '/kaggle/input/blood-vessel-segmentation'
OUTPUT_DIR = '/kaggle/working/output'
EPOCHS = 50

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def train_model(resume_from=None):
  print("GPU Memory Usage:")
  try:
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
      device_name = device.name.split(':')[-2] + ':' + device.name.split(':')[-1]
      memory_info = tf.config.experimental.get_memory_info(device_name)
      print(f"Current memory usage for {device_name}: {memory_info['current'] / 1e9:.2f} GB")
      print(f"Peak memory usage for {device_name}: {memory_info['peak'] / 1e9:.2f} GB")
  except Exception as e:
    print(f"Could not get GPU memory info: {str(e)}")
    print("Continuing with training...")

  print("Starting training...")
  preprocessor = DataProcessor(INPUT_DIR)
  image_paths, label_paths = preprocessor.get_file_paths('train')


  # Set seed for reproducibility
