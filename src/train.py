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

  split_idx = int(len(image_paths) * 0.8)
  train_generator = MemoryEfficientSequence(
    image_paths[:split_idx],
    label_paths[:split_idx]
  )
  val_generator = MemoryEfficientSequence(
    image_paths[split_idx:],
    label_paths[split_idx:]
  )

  model = create_memory_efficient_unet(input_shape=(16, 64, 64, 1))

  optimizer = tf.keras.mizxed_precision.LossScaleOptimizer(
		tf.keras.optimizers.Adam(learning_rate=1e-4)
	)

  model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', MetricsCollection.dice_coefficient,
             MetricsCollection.precision,
             MetricsCollection.recall]
  )

  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  save_dir = f'{OUTPUT_DIR}/gpu_model_{timestamp}'
  os.makedirs(save_dir, exist_ok=True)

  initial_epoch = 0
  if resume_from:
    print(f"Loading weights from {resume_from}")
    model.load_weights(resume_from)
    try:
      initial_epoch = int(resume_from.split('_')[-1].split('.')[0])
    except:
      pass

    with open(f'{save_dir}/model_summary.txt', 'w') as f:
      model.summary(print_fn=lambda x: f.write(x + '\n'))

    callbacks = [
      # save model after each epoch
      tf.keras.callbacks.ModelCheckpoint(
        f'{save_dir}/model_epoch_{{epoch:03d}}.keras',
        save_best_only=False
			),
      tf.keras.callbacks.CSVLogger(f'{save_dir}/training_log.csv'),
      tf.keras.callbacks.EarlyStopping(
        monitor='val_dice',
        patience=10,
        restore_best_weights=True
      ),
      tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_dice_coefficient',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        node='max'
      ),
      MemoryCleanupCallback(PerformanceMonitor())
	]

    
  # Set seed for reproducibility
