import os
import tensorflow as tf


# Enable TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# Check if TensorFlow is built with CUDA support
print("Is TensorFlow built with CUDA support? ", tf.test.is_built_with_cuda())

# List physical devices
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU is available.")
    for gpu in gpus:
        print("TensorFlow is using GPU:", gpu)
else:
    print("No GPU available. TensorFlow is running on CPU.")