import os
import gc
import time
import psutil
import tensorflow as tf
from tensorflow.keras import backend as K

class PerformanceMonitor:
  def __init__(self):
    self.start_time = time.time()

  def log_memory_usage(self):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory Usage: {memory_info.rss / 1024 / 1024:.2f} MB")

  def log_time_elapsed(self):
    elapsed_time = time.time() - self.start_time
    print(f"Time Elapsed: {elapsed_time:.2f} seconds")

class MemoryCleanupCallback(tf.keras.callbacks.Callback):
  def __init__(self, monitor):
    super().__init__()
    self.monitor = monitor

  def on_epoch_end(self, epoch, logs=None):
    gc.collect()
    K.clear_session()
    self.monitor.log_memory_usage()
    print(f"\nMemory cleaned after epoch {epoch}")

  def on_batch_end(self, batch, logs=None):
    if batch % 100 == 0:
      gc.collect()

