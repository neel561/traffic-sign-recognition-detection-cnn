import base64
import io
import mimetypes
import sys
from pathlib import Path

from flask import Flask, jsonify, request
from PIL import Image

# Make "import tsr" work however the app is started (python algo/Traffic_app.py, flask --app algo.Traffic_app, ...)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect as detection  # noqa: E402
import tsr  # noqa: E402

MAX_UPLOAD_MB = 20

# Older Pythons on Windows don't know .webp, which the Prediction page background uses
mimetypes.add_type('image/webp', '.webp')

# The website is plain HTML, so serve that folder as static files from the site root
app = Flask(__name__, static_folder=str(tsr.ALGO_DIR.parent / "website"), static_url_path="")
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024

# Load the model once at startup rather than on every request
model = tsr.load_model()
detector = None


def get_detector():
    """Load the sign detector the first time a photo is sent to /detect."""
    global detector
    if detector is None:
        detector = detection.load_detector()
    return detector


def uploaded_image():
    """Return (image, error response); exactly one of the two is None."""
    file = request.files.get('file')
    if file is None or file.filename == '':
        return None, (jsonify(error="No image was uploaded"), 400)
    try:
        return tsr.open_image(file.stream), None
    except (OSError, Image.DecompressionBombError):
        # OSError also covers UnidentifiedImageError, and truncated or corrupt image files
        return None, (jsonify(error="That file could not be read as an image"), 400)


@app.route('/')
def index():
    return app.send_static_file('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Name the sign in a picture that is already cropped around one sign."""
    image, error = uploaded_image()
    if error:
        return error
    class_id, sign, confidence = tsr.predict(model, image)
    return jsonify(class_id=class_id, sign=sign, confidence=confidence,
                   recognised=confidence >= tsr.MIN_CONFIDENCE)


@app.route('/detect', methods=['POST'])
def detect():
    """Find every traffic sign in a photo and name each one."""
    if not detection.DETECTOR_PATH.exists():
        return jsonify(error="The sign detector has not been trained yet. Run: python algo/train_detector.py"), 503
    image, error = uploaded_image()
    if error:
        return error
    signs = detection.detect(get_detector(), model, image)
    annotated = detection.annotate(image, signs)
    annotated.thumbnail((1024, 1024))
    buffer = io.BytesIO()
    annotated.save(buffer, "JPEG", quality=85)
    return jsonify(signs=signs,
                   image="data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode())


@app.errorhandler(413)
def upload_too_large(error):
    return jsonify(error=f"That image is too large (the limit is {MAX_UPLOAD_MB} MB)"), 413


if __name__ == '__main__':
    app.run()
