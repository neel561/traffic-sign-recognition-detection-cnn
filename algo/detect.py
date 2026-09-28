"""Find the traffic signs in a photo and name each one.

Two models work together here. A YOLO detector, trained on the GTSDB road photos, draws a box
around every traffic sign it can find. Each box is then cropped out and passed to the CNN in
tsr.py, which says which of the 43 signs it is.
"""
import argparse

from PIL import Image, ImageDraw, ImageFont

import tsr

DETECTOR_PATH = tsr.ALGO_DIR / "model" / "sign_detector.pt"
MIN_BOX_CONFIDENCE = 0.25  # how sure the detector must be that a box holds a sign
BOX_PADDING = 0.15         # grow each box by this much, so the crops look like our training images
IMAGE_SIZE = 1024          # the size the detector was trained at; smaller misses distant signs


def load_detector(path=DETECTOR_PATH):
    import torch
    from ultralytics import YOLO, settings  # imported here because it is slow to load
    settings.update({"sync": False})  # don't send usage analytics
    detector = YOLO(str(path))
    if torch.cuda.is_available():
        detector.to("cuda")
    return detector


def pad_box(box, size, padding=BOX_PADDING):
    left, top, right, bottom = box
    grow_x, grow_y = (right - left) * padding, (bottom - top) * padding
    return (max(0, left - grow_x), max(0, top - grow_y),
            min(size[0], right + grow_x), min(size[1], bottom + grow_y))


def detect(detector, classifier, image, min_box_confidence=MIN_BOX_CONFIDENCE, padding=BOX_PADDING):
    """Return one entry per sign found, with its box, name and both models' confidence."""
    if not isinstance(image, Image.Image):
        image = tsr.open_image(image)
    image = image.convert("RGB")
    boxes = detector.predict(image, conf=min_box_confidence, imgsz=IMAGE_SIZE, verbose=False)[0].boxes
    found = boxes.xyxy.tolist()
    crops = [image.crop(pad_box(box, image.size, padding)) for box in found]
    signs = []
    for box, box_confidence, (class_id, sign, confidence) in zip(found, boxes.conf.tolist(),
                                                                 tsr.predict_many(classifier, crops)):
        signs.append({
            "box": [round(value) for value in box],
            "class_id": class_id,
            "sign": sign,
            "confidence": confidence,
            "box_confidence": box_confidence,
            "recognised": confidence >= tsr.MIN_CONFIDENCE,
        })
    return signs


def annotate(image, signs):
    """Draw the boxes and sign names on a copy of the photo."""
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(14, image.width // 60))
    for sign in signs:
        colour = "#32cd32" if sign["recognised"] else "#ffa500"
        label = (f"{sign['sign']} {sign['confidence']:.0%}" if sign["recognised"]
                 else f"sign? (best guess {sign['sign']})")
        left, top, right, bottom = sign["box"]
        draw.rectangle((left, top, right, bottom), outline=colour, width=3)
        text_left, text_top, text_right, text_bottom = draw.textbbox((left + 4, top + 3), label, font=font)
        draw.rectangle((left, top, text_right + 4, text_bottom + 3), fill=colour)
        draw.text((left + 4, top + 3), label, fill="black", font=font)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photo", help="Photo to look for traffic signs in")
    parser.add_argument("--out", help="Where to save the photo with the boxes drawn on it")
    parser.add_argument("--min-box-confidence", type=float, default=MIN_BOX_CONFIDENCE)
    args = parser.parse_args()

    image = tsr.open_image(args.photo)
    signs = detect(load_detector(), tsr.load_model(), image, args.min_box_confidence)
    for sign in signs:
        state = sign["sign"] if sign["recognised"] else f"not recognised (best guess {sign['sign']})"
        print(f"{tuple(sign['box'])}  {state}  sign {sign['box_confidence']:.0%} / class {sign['confidence']:.0%}")
    print(f"{len(signs)} sign(s) found")
    if args.out:
        annotate(image, signs).save(args.out)
        print("saved", args.out)


if __name__ == "__main__":
    main()
