"""Find and name traffic signs in a video or a live webcam feed.

    python algo/video.py                 # webcam
    python algo/video.py clip.mp4        # a video file
    python algo/video.py clip.mp4 --save out.mp4

Press q to close the window. Each frame goes through the same two steps as detect.py: the detector
finds the signs, then the CNN names each one.
"""
import argparse
import time

import cv2
import numpy as np
from PIL import Image

import tsr
from detect import MIN_BOX_CONFIDENCE, annotate, detect, load_detector


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", nargs="?", help="Video file. Leave out to use the webcam")
    parser.add_argument("--camera", type=int, default=0, help="Which webcam to use")
    parser.add_argument("--save", help="Also write the annotated video to this file")
    parser.add_argument("--no-window", action="store_true", help="Don't open a window; useful with --save")
    parser.add_argument("--min-box-confidence", type=float, default=MIN_BOX_CONFIDENCE)
    args = parser.parse_args()

    detector, classifier = load_detector(), tsr.load_model()
    capture = cv2.VideoCapture(args.video if args.video else args.camera)
    if not capture.isOpened():
        raise SystemExit(f"Could not open {args.video or 'the webcam'}")

    writer = None
    if args.save:
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                 capture.get(cv2.CAP_PROP_FPS) or 25, size)

    frames, signs_seen, started = 0, 0, time.perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        # OpenCV gives frames as BGR; our models work on RGB
        rgb = Image.fromarray(frame[:, :, ::-1])
        signs = detect(detector, classifier, rgb, args.min_box_confidence)
        annotated = np.array(annotate(rgb, signs))[:, :, ::-1]
        frames, signs_seen = frames + 1, signs_seen + len(signs)

        if not args.no_window:
            cv2.imshow("Traffic sign detection - press q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if writer:
            writer.write(annotated)

    capture.release()
    if writer:
        writer.release()
        print("saved", args.save)
    cv2.destroyAllWindows()
    seconds = time.perf_counter() - started
    print(f"{frames} frames in {seconds:.1f}s ({frames / seconds:.1f} per second), {signs_seen} sign(s) seen")


if __name__ == "__main__":
    main()
