
import tensorflow as tf
import numpy as np

traing_Data = tf.keras.preprocessing.image_dataset_from_directory(
    "archive/chest_xray/chest_xray/train",
    batch_size=8,
    color_mode="grayscale"
)

# Val_data = tf.keras.preprocessing.image_dataset_from_directory(
#     "archive/chest_xray/chest_xray/val",
#     image_size=(128,128),
#     batch_size=8
# )

test_Data = tf.keras.preprocessing.image_dataset_from_directory(
    "archive/chest_xray/chest_xray/test",
    image_size=(128,128),
    batch_size=8
)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

traing_Data = traing_Data.map(
    lambda x, y: (data_augmentation(x, training=True), y)
)

datasets = {
    0: test_Data,
    1: Val_data,
    2: traing_Data
}




# Cache to store computed results
_image_cache = {}

def get_images_and_labels(number):
    # Check if result is already in cache
    if number in _image_cache:
        return _image_cache[number]
    
    dataset = datasets.get(number, None)
    
    if dataset:
        images = []
        labels = []
        
        for image_batch, label_batch in dataset:
            images.append(image_batch)
            labels.append(label_batch)
        
        # Concatenate all batches into single arrays
        images = tf.concat(images, axis=0)
        labels = tf.concat(labels, axis=0)
        
        # Cache the result
        _image_cache[number] = (images, labels)
        
        return images, labels
    else:
        return None, None



def clear_cache():
    """
    Clear the in-memory cache if needed
    """
    global _image_cache
    _image_cache.clear()

# Optional: Add a way to manually set cache
def set_cached_dataset(number, images, labels):
    """
    Manually set a cached dataset
    
    Args:
        number (int): Dataset identifier
        images (tf.Tensor): Cached images
        labels (tf.Tensor): Cached labels
    """
    _image_cache[number] = (images, labels)

# Optionally, you can add a method to check cache
def is_cached(number):
    """
    Check if a dataset is already cached
    
    Args:
        number (int): Dataset identifier
    
    Returns:
        bool: Whether the dataset is cached
    """
    return number in _image_cache




def load_dataset(number):
    images, labels = get_images_and_labels(number)
    return images, labels