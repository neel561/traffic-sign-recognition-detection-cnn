import mimetypes
import sys
from pathlib import Path

from flask import Flask, jsonify, request
from PIL import Image

# Make "import tsr" work however the app is started (python algo/Traffic_app.py, flask --app algo.Traffic_app, ...)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tsr  # noqa: E402

MAX_UPLOAD_MB = 20

# Older Pythons on Windows don't know .webp, which the Prediction page background uses
mimetypes.add_type('image/webp', '.webp')

# The website is plain HTML, so serve that folder as static files from the site root
app = Flask(__name__, static_folder=str(tsr.ALGO_DIR.parent / "website"), static_url_path="")
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024

# Load the model once at startup rather than on every request
model = tsr.load_model()


@app.route('/')
def index():
    return app.send_static_file('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if file is None or file.filename == '':
        return jsonify(error="No image was uploaded"), 400
    try:
        class_id, sign, confidence = tsr.predict(model, file.stream)
    except (OSError, Image.DecompressionBombError):
        # OSError also covers UnidentifiedImageError, and truncated or corrupt image files
        return jsonify(error="That file could not be read as an image"), 400
    return jsonify(class_id=class_id, sign=sign, confidence=confidence,
                   recognised=confidence >= tsr.MIN_CONFIDENCE)


@app.errorhandler(413)
def upload_too_large(error):
    return jsonify(error=f"That image is too large (the limit is {MAX_UPLOAD_MB} MB)"), 413


if __name__ == '__main__':
    app.run()
