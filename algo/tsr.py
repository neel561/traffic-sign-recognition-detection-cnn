"""Shared code for the traffic sign recognition model: class names, image loading, preprocessing and prediction."""
from pathlib import Path

import numpy as np
from keras.models import load_model as keras_load_model
from PIL import Image, ImageOps

ALGO_DIR = Path(__file__).resolve().parent
MODEL_PATH = ALGO_DIR / "model" / "TSR.keras"
IMG_SIZE = (30, 30)

# Below this confidence the apps say no sign was recognised. On the GTSRB test set this rejects 3% of
# images, which include 40% of the model's mistakes; blank or noise images score about 51%.
MIN_CONFIDENCE = 0.6

# Classes of traffic signs
CLASSES = {0: 'Speed limit (20km/h)',
           1: 'Speed limit (30km/h)',
           2: 'Speed limit (50km/h)',
           3: 'Speed limit (60km/h)',
           4: 'Speed limit (70km/h)',
           5: 'Speed limit (80km/h)',
           6: 'End of speed limit (80km/h)',
           7: 'Speed limit (100km/h)',
           8: 'Speed limit (120km/h)',
           9: 'No passing',
           10: 'No passing veh over 3.5 tons',
           11: 'Right-of-way at intersection',
           12: 'Priority road',
           13: 'Yield',
           14: 'Stop',
           15: 'No vehicles',
           16: 'Vehicle > 3.5 tons prohibited',
           17: 'No entry',
           18: 'General caution',
           19: 'Dangerous curve left',
           20: 'Dangerous curve right',
           21: 'Double curve',
           22: 'Bumpy road',
           23: 'Slippery road',
           24: 'Road narrows on the right',
           25: 'Road work',
           26: 'Traffic signals',
           27: 'Pedestrians',
           28: 'Children crossing',
           29: 'Bicycles crossing',
           30: 'Beware of ice/snow',
           31: 'Wild animals crossing',
           32: 'End speed + passing limits',
           33: 'Turn right ahead',
           34: 'Turn left ahead',
           35: 'Ahead only',
           36: 'Go straight or right',
           37: 'Go straight or left',
           38: 'Keep right',
           39: 'Keep left',
           40: 'Roundabout mandatory',
           41: 'End of no passing',
           42: 'End no passing vehicle > 3.5 tons'}


def open_image(source):
    """Open an image (path or file object) the right way up, using its EXIF orientation tag.

    Phone photos are often stored sideways with a tag saying how to rotate them. The pixels are
    decoded here too, so a truncated or corrupt file raises OSError straight away.
    """
    image = Image.open(source)
    image.load()
    return ImageOps.exif_transpose(image)


def preprocess(image):
    """Turn an image (path, file object or PIL image) into the 30x30 RGB array the model uses."""
    if not isinstance(image, Image.Image):
        image = open_image(image)
    # convert() handles PNGs with transparency or palettes, which would otherwise not be 3-channel
    return np.array(image.convert("RGB").resize(IMG_SIZE))


def load_model(path=MODEL_PATH):
    return keras_load_model(path)


def predict(model, image):
    """Return (class_id, sign_name, confidence) for a single image."""
    return predict_many(model, [image])[0]


def predict_many(model, images):
    """The same as predict, but for several images in one go.

    Calling the model once for the whole batch is far quicker than calling model.predict() for each
    image, which matters when a photo holds several signs, and on video.
    """
    if not images:
        return []
    batch = np.array([preprocess(image) for image in images])
    probabilities = np.asarray(model(batch, training=False))
    return [(int(row.argmax()), CLASSES[int(row.argmax())], float(row.max())) for row in probabilities]
