import os
import gc
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision
from datetime import datetime

from .model import create_memory_efficient_unet3d
from .data import DataPreprocessor, MemoryEfficientSequence
from .metrics import MetricsCollection
from .utils import PerformanceMonitor, MemoryCleanupCallback

# Enable mixed precision for better GPU performance (TensorFlow on CPU does not support mixed precision)
# mixed_precision.set_global_policy('mixed_float16')

# Constants
BATCH_SIZE = 8
VALIDATION_SPLIT = 0.2
WORKING_DIR = '/kaggle/working'
INPUT_DIR = 'data'
OUTPUT_DIR = 'output'
EPOCHS = 50

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

    print("Starting training process...")
    preprocessor = DataPreprocessor(INPUT_DIR)
    image_paths, label_paths = preprocessor.get_file_paths('train')

    # split_idx = int(len(image_paths) * 0.8)
    # train_generator = MemoryEfficientSequence(
    #     image_paths[:split_idx],
    #     label_paths[:split_idx]
    # )
    # val_generator = MemoryEfficientSequence(
    #     image_paths[split_idx:],
    #     label_paths[split_idx:]
    # )
    train_generator = MemoryEfficientSequence(image_paths, label_paths)
    val_generator = None

    input_shape = (16, 64, 64, 1)
    model = create_memory_efficient_unet3d(input_shape)

    # Use mixed precision optimizer
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
        tf.keras.optimizers.Adam(learning_rate=5e-4)
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', MetricsCollection.dice_coefficient,
                MetricsCollection.precision, MetricsCollection.recall]
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f'{OUTPUT_DIR}/gpu_model_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)

    initial_epoch = 0
    if resume_from:
        print(f"Loading weights from {resume_from}")
        model.load_weights(resume_from)
        try:
            initial_epoch = int(resume_from.split('epoch')[-1].split('.')[0])
        except:
            pass

    with open(f'{save_dir}/model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    callbacks = [
        # Save model after each epoch
        tf.keras.callbacks.ModelCheckpoint(
            f'{save_dir}/model_epoch{{epoch:03d}}.keras',
            save_best_only=False
        ),
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            f'{save_dir}/best_model.keras',
            save_best_only=True,
            monitor='val_dice_coefficient',
            mode='max'
        ),
        tf.keras.callbacks.CSVLogger(f'{save_dir}/training_log.csv'),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_dice_coefficient',
            patience=5,
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_dice_coefficient',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            mode='max'
        ),
        MemoryCleanupCallback(PerformanceMonitor())
    ]

    try:
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            initial_epoch=initial_epoch,
            epochs=EPOCHS,
            callbacks=callbacks
        )

        pd.DataFrame(history.history).to_csv(f'{save_dir}/history.csv')
        model.save(f'{save_dir}/final_model.keras')

        return model, history, save_dir

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state...")
        model.save(f'{save_dir}/interrupted_model.keras')
        raise

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    try:
        # Set memory growth for GPU
        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

        model, history, model_dir = train_model()
        print(f"\nTraining complete. Model and outputs saved to: {model_dir}")
        print("\nFinal metrics:")
        print(pd.DataFrame(history.history).tail(1))
        monitor.log_memory_usage()
        monitor.log_time_elapsed()

    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        print("Cleaning up final resources...")
        gc.collect()
        tf.keras.backend.clear_session()
