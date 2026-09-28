"""Train the traffic sign detector on the GTSDB photos prepared by prepare_gtsdb.py.

    python algo/prepare_gtsdb.py --zip FullIJCNN2013.zip
    python algo/train_detector.py

The detector only learns to find signs, not to tell them apart: the CNN in tsr.py names each one.
The best weights are copied to algo/model/sign_detector.pt, which detect.py loads.

Nothing here imports tsr or detect on purpose. YOLO's data loader starts worker processes that
re-import this file, and loading TensorFlow in every worker would use gigabytes of memory.
"""
import argparse
import shutil
from pathlib import Path

ALGO_DIR = Path(__file__).resolve().parent
DETECTOR_PATH = ALGO_DIR / "model" / "sign_detector.pt"  # keep in step with detect.DETECTOR_PATH
DATA = ALGO_DIR.parent / "Dataset" / "GTSDB" / "gtsdb.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained model to start from")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024, help="Signs are small, so we train at a high resolution")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="'0' for the first GPU, 'cpu' to force the processor")
    args = parser.parse_args()

    from ultralytics import YOLO, settings
    settings.update({"sync": False})  # don't send usage analytics
    model = YOLO(args.model)
    results = model.train(data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                          workers=args.workers, device=args.device, seed=0,
                          project=str(ALGO_DIR.parent / "runs"), name="sign_detector", exist_ok=True)

    DETECTOR_PATH.parent.mkdir(exist_ok=True)
    shutil.copy(Path(results.save_dir) / "weights" / "best.pt", DETECTOR_PATH)
    print("saved detector to", DETECTOR_PATH)


if __name__ == "__main__":
    main()
