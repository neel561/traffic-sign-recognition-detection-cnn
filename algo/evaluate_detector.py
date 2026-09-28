"""Measure how well the detector and the classifier work together on the GTSDB test photos.

    python algo/evaluate_detector.py

It reports two things. First, how many of the signs in the 300 test photos the detector finds (a
box counts as a find when it overlaps the real box by at least half). Second, how many of those
found signs the CNN then names correctly. Results are written to algo/results/detection_metrics.txt.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import tsr
from detect import MIN_BOX_CONFIDENCE, detect, load_detector
from tsr import ALGO_DIR

GTSDB = ALGO_DIR.parent / "Dataset" / "GTSDB"
RESULTS = ALGO_DIR / "results" / "detection_metrics.txt"
FIRST_TEST_PHOTO = 600
IOU_MATCH = 0.5


def overlap(box, other):
    """Intersection over union: how much two boxes overlap, from 0 (not at all) to 1 (exactly)."""
    left, top, right, bottom = (max(box[0], other[0]), max(box[1], other[1]),
                                min(box[2], other[2]), min(box[3], other[3]))
    if right <= left or bottom <= top:
        return 0.0
    shared = (right - left) * (bottom - top)
    area = (box[2] - box[0]) * (box[3] - box[1]) + (other[2] - other[0]) * (other[3] - other[1])
    return shared / (area - shared)


def read_ground_truth(gt_file):
    """{photo stem: [(box, class id), ...]} for the test photos only."""
    signs = defaultdict(list)
    for line in Path(gt_file).read_text().splitlines():
        if not line.strip():
            continue
        name, left, top, right, bottom, class_id = line.split(";")
        stem = Path(name).stem
        if int(stem) >= FIRST_TEST_PHOTO:
            signs[stem].append(((int(left), int(top), int(right), int(bottom)), int(class_id)))
    return signs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photos", type=Path, default=GTSDB / "images" / "val")
    parser.add_argument("--ground-truth", type=Path, default=GTSDB / "raw" / "FullIJCNN2013" / "gt.txt")
    parser.add_argument("--min-box-confidence", type=float, default=MIN_BOX_CONFIDENCE)
    args = parser.parse_args()

    truth = read_ground_truth(args.ground_truth)
    detector, classifier = load_detector(), tsr.load_model()

    found = named = missed = false_boxes = total = 0
    for photo in sorted(args.photos.glob("*.jpg")):
        actual = list(truth.get(photo.stem, []))
        total += len(actual)
        matched = set()
        for sign in sorted(detect(detector, classifier, photo, args.min_box_confidence),
                           key=lambda s: -s["box_confidence"]):
            best, best_overlap = None, IOU_MATCH
            for index, (box, class_id) in enumerate(actual):
                if index not in matched and overlap(sign["box"], box) >= best_overlap:
                    best, best_overlap = index, overlap(sign["box"], box)
            if best is None:
                false_boxes += 1
                continue
            matched.add(best)
            found += 1
            named += sign["recognised"] and sign["class_id"] == actual[best][1]
        missed += len(actual) - len(matched)

    photos = len(list(args.photos.glob("*.jpg")))
    report = (
        f"GTSDB test photos: {photos}, signs in them: {total}\n"
        f"Detector confidence threshold: {args.min_box_confidence}\n"
        f"Signs found (box overlap >= {IOU_MATCH}): {found}/{total} = {found / total:.1%}\n"
        f"Signs missed:                            {missed}\n"
        f"Boxes that were not signs:               {false_boxes} ({false_boxes / photos:.2f} per photo)\n"
        f"Found and named correctly:               {named}/{total} = {named / total:.1%} of all signs, "
        f"{named / found:.1%} of the ones found\n"
    )
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
