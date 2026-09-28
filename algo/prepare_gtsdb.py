"""Turn the GTSDB detection dataset into the folder layout YOLO expects.

GTSDB (German Traffic Sign Detection Benchmark) is 900 road photos of 1360x800 pixels with a box
drawn around every traffic sign. Download FullIJCNN2013.zip (1.7 GB) from
https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip
and point this script at it:

    python algo/prepare_gtsdb.py --zip path/to/FullIJCNN2013.zip

The photos are converted to JPEG and the boxes are written as YOLO labels, using a single class
("traffic sign"): the detector only finds signs, and our CNN says which sign each one is. The
standard GTSDB split is used, the first 600 photos for training and the last 300 for testing.
"""
import argparse
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from tsr import ALGO_DIR

DEFAULT_OUT = ALGO_DIR.parent / "Dataset" / "GTSDB"
TRAIN_IMAGES = 600  # photos 00000-00599 are the training set, 00600-00899 the test set


def extract(zip_path, target):
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target)
    source = target / "FullIJCNN2013"
    return source if source.is_dir() else target


def read_boxes(source):
    """Read gt.txt into {photo file name: [(left, top, right, bottom), ...]}."""
    boxes = defaultdict(list)
    for line in (source / "gt.txt").read_text().splitlines():
        if not line.strip():
            continue
        name, left, top, right, bottom, _class_id = line.split(";")
        boxes[name].append((int(left), int(top), int(right), int(bottom)))
    return boxes


def convert(photo, boxes, out, split):
    image = Image.open(photo)
    width, height = image.size
    image.convert("RGB").save(out / "images" / split / f"{photo.stem}.jpg", quality=95)
    lines = []
    for left, top, right, bottom in boxes:
        # YOLO wants the box centre, width and height, each as a fraction of the photo size
        lines.append(f"0 {(left + right) / 2 / width:.6f} {(top + bottom) / 2 / height:.6f} "
                     f"{(right - left) / width:.6f} {(bottom - top) / height:.6f}")
    (out / "labels" / split / f"{photo.stem}.txt").write_text("\n".join(lines))
    return [(right - left) for left, _, right, _ in boxes]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, help="FullIJCNN2013.zip")
    parser.add_argument("--source", type=Path, help="Folder with the already unzipped photos and gt.txt")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.zip and not args.source:
        parser.error("give either --zip or --source")

    for split in ("train", "val"):
        for kind in ("images", "labels"):
            (args.out / kind / split).mkdir(parents=True, exist_ok=True)

    source = args.source if args.source else extract(args.zip, args.out / "raw")
    boxes = read_boxes(source)
    photos = sorted(source.glob("*.ppm"))
    print(f"{len(photos)} photos, {sum(len(b) for b in boxes.values())} signs")

    widths = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for sign_widths in pool.map(
                lambda photo: convert(photo, boxes.get(photo.name, []), args.out,
                                      "train" if int(photo.stem) < TRAIN_IMAGES else "val"),
                photos):
            widths += sign_widths

    (args.out / "gtsdb.yaml").write_text(
        f"path: {args.out.as_posix()}\ntrain: images/train\nval: images/val\n\nnames:\n  0: traffic sign\n")
    empty = sum(1 for photo in photos if not boxes.get(photo.name))
    print(f"train {len(list((args.out / 'images' / 'train').glob('*.jpg')))} photos, "
          f"val {len(list((args.out / 'images' / 'val').glob('*.jpg')))} photos, "
          f"{empty} photos with no sign")
    print(f"sign widths: smallest {min(widths)} px, median {sorted(widths)[len(widths) // 2]} px, largest {max(widths)} px")
    print("wrote", args.out / "gtsdb.yaml")


if __name__ == "__main__":
    main()
