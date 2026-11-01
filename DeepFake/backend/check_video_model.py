"""
Video model sanity check (roadmap T0.4).

Answers one question: how must face images be prepared, and how must the
model's output be read, for the deployed video model to give correct answers?

The training script (train_model.py) feeds RGB images through Xception's
preprocess_input (pixels scaled to -1..1) with classes ['fake', 'real'].
app.py feeds BGR images scaled to 0..1 and treats output[0][0] <= 0.5 as fake.
At most one of those can match the model. This script scores faces you know
the answer for under every combination and prints which one is right.

Usage (from DeepFake/backend, with the backend's virtual environment active):

    python check_video_model.py --real path/to/real --fake path/to/fake

Each folder can hold videos (.mp4 .avi .mov .mkv) and/or images (.jpg .png ...).
Faces are found with dlib exactly as app.py does. Use at least 10 real and 10
fake videos (or 20+ face images each) from a source the model was NOT trained
on if you can. Results are printed and saved to check_video_model_report.json.

Nothing in the app is changed by running this.
"""

import argparse
import json
import os
import sys

import numpy as np

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Ways of preparing a face crop. cv2 gives BGR pixels 0..255.
#   bgr_01   : what app.py does today (BGR, divided by 255)
#   rgb_xcep : what train_model.py does (RGB, Xception preprocess_input -> -1..1)
#   rgb_01   : RGB, divided by 255
#   bgr_xcep : BGR, scaled to -1..1
MODES = {
    "bgr_01": lambda bgr: bgr / 255.0,
    "rgb_xcep": lambda bgr: bgr[..., ::-1] / 127.5 - 1.0,
    "rgb_01": lambda bgr: bgr[..., ::-1] / 255.0,
    "bgr_xcep": lambda bgr: bgr / 127.5 - 1.0,
}


def load_video_model():
    """Loads the same model file app.py uses."""
    from huggingface_hub import hf_hub_download
    from tensorflow.keras.models import load_model

    path = hf_hub_download(repo_id="Codesmith-23/deepfake-detector-v1", filename="model_fine_final.h5")
    print(f"Model file: {path}")
    return load_model(path)


def describe_model(model):
    last = model.layers[-1]
    activation = getattr(getattr(last, "activation", None), "__name__", "unknown")
    info = {
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
        "last_layer": type(last).__name__,
        "last_activation": activation,
    }
    for key, value in info.items():
        print(f"  {key:16} {value}")
    return info


def face_crops_from_frame(frame, detector):
    """Face crops from one BGR frame, cropped exactly like app.py."""
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    crops = []
    for d in detector(gray, 0):
        x1, y1 = max(0, d.left()), max(0, d.top())
        x2, y2 = min(frame.shape[1], d.right()), min(frame.shape[0], d.bottom())
        if x2 - x1 >= 10 and y2 - y1 >= 10:
            crops.append(frame[y1:y2, x1:x2])
    return crops


def collect_faces(folder, detector, faces_per_video):
    """Returns a list of (source_name, bgr_crop) for every video and image in folder."""
    import cv2

    faces = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        ext = os.path.splitext(name)[1].lower()
        if ext in VIDEO_EXT:
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            found = 0
            # Spread the sampled frames over the whole video.
            for frame_no in np.linspace(0, total - 1, faces_per_video * 3).astype(int):
                if found >= faces_per_video:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
                ok, frame = cap.read()
                if not ok:
                    continue
                for crop in face_crops_from_frame(frame, detector)[:1]:
                    faces.append((name, crop))
                    found += 1
            cap.release()
            print(f"  {name}: {found} faces")
        elif ext in IMAGE_EXT:
            img = cv2.imread(path)
            if img is None:
                continue
            crops = face_crops_from_frame(img, detector)
            # An image with no detectable face is assumed to already be a face crop.
            faces.append((name, crops[0] if crops else img))
    return faces


def score_faces(model, faces, input_size):
    """Runs every face through the model under every mode. Returns {mode: outputs array (n, k)}."""
    import cv2

    resized = np.array([cv2.resize(crop, input_size).astype(np.float32) for _, crop in faces])
    results = {}
    for mode, prepare in MODES.items():
        out = model.predict(prepare(resized), verbose=0, batch_size=16)
        results[mode] = np.asarray(out, dtype=np.float64).reshape(len(faces), -1)
    return results


def evaluate(outputs, is_fake):
    """
    For one mode, tries each reading of the output and returns the accuracy of each.
    is_fake: boolean array, the true answer per face.
    """
    readings = {}
    for idx in range(outputs.shape[1]):
        p = outputs[:, idx]
        # "output[idx] is P(fake)": fake when > 0.5
        readings[f"out[{idx}] > 0.5 means fake"] = float(np.mean((p > 0.5) == is_fake))
        # "output[idx] is P(real)": fake when <= 0.5  (app.py today uses idx 0 this way)
        readings[f"out[{idx}] <= 0.5 means fake"] = float(np.mean((p <= 0.5) == is_fake))
    return readings


def video_level_accuracy(outputs, is_fake, sources, idx, fake_when_high):
    """Accuracy per video/image (average over its faces), closer to what users see."""
    correct = []
    for src in sorted(set(sources)):
        rows = [i for i, s in enumerate(sources) if s == src]
        p = outputs[rows, idx].mean()
        predicted_fake = p > 0.5 if fake_when_high else p <= 0.5
        correct.append(predicted_fake == is_fake[rows[0]])
    return float(np.mean(correct)), len(correct)


def video_scores(outputs, sources, idx):
    """Average output per video/image, like app.py's avg_prob. Returns (names, scores)."""
    names = sorted(set(sources))
    scores = np.array([outputs[[i for i, s in enumerate(sources) if s == n], idx].mean() for n in names])
    return names, scores


def accuracy_at(scores, is_fake, threshold, fake_when_high):
    predicted_fake = scores > threshold if fake_when_high else scores <= threshold
    return float(np.mean(predicted_fake == is_fake))


def best_threshold(scores, is_fake, fake_when_high, candidates):
    """Highest-accuracy threshold; ties go to the one closest to 0.5."""
    return max(candidates, key=lambda t: (accuracy_at(scores, is_fake, t, fake_when_high), -abs(t - 0.5)))


def threshold_sweep(outputs, is_fake_faces, sources, idx, fake_when_high):
    """
    Finds the cut-off that gets the most videos right, and estimates honestly how
    well it would do on new videos (leave-one-video-out: pick the threshold
    without the video, then test on it).
    """
    names, scores = video_scores(outputs, sources, idx)
    is_fake = np.array([n.startswith("fake/") for n in names])
    candidates = [round(t, 2) for t in np.arange(0.05, 0.96, 0.01)]

    print("\nThreshold sweep (video level, the verdict users see):")
    for t in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7):
        print(f"  cut-off {t:.2f}: {accuracy_at(scores, is_fake, t, fake_when_high) * 100:5.1f}% of videos")

    best = best_threshold(scores, is_fake, fake_when_high, candidates)
    best_acc = accuracy_at(scores, is_fake, best, fake_when_high)
    now_acc = accuracy_at(scores, is_fake, 0.5, fake_when_high)

    held_out_correct = []
    for i in range(len(names)):
        keep = np.arange(len(names)) != i
        t = best_threshold(scores[keep], is_fake[keep], fake_when_high, candidates)
        held_out_correct.append(accuracy_at(scores[i:i + 1], is_fake[i:i + 1], t, fake_when_high))
    honest = float(np.mean(held_out_correct))

    print(f"\n  Today (0.50):        {now_acc * 100:.1f}% of {len(names)} videos")
    print(f"  Best cut-off ({best:.2f}): {best_acc * 100:.1f}% on these videos")
    print(f"  Expected on new videos with tuning: {honest * 100:.1f}% (leave-one-out)")
    if honest - now_acc >= 0.05:
        verdict = "change"
        print(f"  Worth changing: set the threshold in app.py to {best:.2f}.")
    else:
        verdict = "keep"
        print("  Not worth changing: the gain does not hold up on held-out videos. Keep 0.50.")
    return {"today": round(now_acc, 4), "best_threshold": best, "best_accuracy": round(best_acc, 4),
            "leave_one_out_accuracy": round(honest, 4), "videos": len(names), "verdict": verdict}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", required=True, help="folder with known REAL videos or images")
    parser.add_argument("--fake", required=True, help="folder with known FAKE videos or images")
    parser.add_argument("--faces-per-video", type=int, default=5)
    parser.add_argument("--size", type=int, default=299, help="face size when the model accepts any size (app.py uses 299)")
    parser.add_argument("--report", default="check_video_model_report.json")
    args = parser.parse_args(argv)

    import dlib

    print("Loading model...")
    model = load_video_model()
    print("\nModel:")
    model_info = describe_model(model)
    # The model may accept any size (input_shape (None, None, None, 3)); then use app.py's 299x299.
    h, w = model.input_shape[1:3]
    input_size = (int(w or args.size), int(h or args.size))  # cv2 wants (width, height)
    print(f"  face size used   {input_size[0]}x{input_size[1]}")
    for folder in (args.real, args.fake):
        if not os.path.isdir(folder):
            print(f"ERROR: folder not found: {folder}")
            return 1

    detector = dlib.get_frontal_face_detector()
    print(f"\nFinding faces in REAL folder: {args.real}")
    real = collect_faces(args.real, detector, args.faces_per_video)
    print(f"Finding faces in FAKE folder: {args.fake}")
    fake = collect_faces(args.fake, detector, args.faces_per_video)
    print(f"\n{len(real)} real faces, {len(fake)} fake faces")
    if len(real) < 10 or len(fake) < 10:
        print("WARNING: fewer than 10 faces in a folder; the result will not be reliable.")
    if not real or not fake:
        print("ERROR: need faces in both folders.")
        return 1

    faces = real + fake
    sources = [f"real/{s}" for s, _ in real] + [f"fake/{s}" for s, _ in fake]
    is_fake = np.array([False] * len(real) + [True] * len(fake))

    outputs = score_faces(model, faces, input_size)

    print("\nAverage model output (real faces | fake faces):")
    table = []
    for mode, out in outputs.items():
        means = "  ".join(f"out[{i}] {out[~is_fake, i].mean():.3f} | {out[is_fake, i].mean():.3f}"
                          for i in range(out.shape[1]))
        print(f"  {mode:9} {means}")
        for reading, acc in evaluate(out, is_fake).items():
            table.append((acc, mode, reading))

    table.sort(reverse=True)
    print("\nFace-level accuracy for every combination (50% = guessing):")
    for acc, mode, reading in table:
        marker = "  <- app.py today" if mode == "bgr_01" and reading == "out[0] <= 0.5 means fake" else ""
        print(f"  {acc * 100:5.1f}%  {mode:9} {reading}{marker}")

    best_acc, best_mode, best_reading = table[0]
    best_idx = int(best_reading[4])
    fake_when_high = ">" in best_reading
    vid_acc, n_sources = video_level_accuracy(outputs[best_mode], is_fake, sources, best_idx, fake_when_high)
    current = next(acc for acc, m, r in table if m == "bgr_01" and r == "out[0] <= 0.5 means fake")

    print("\nRESULT")
    print(f"  Best:    {best_mode}, {best_reading}: {best_acc * 100:.1f}% of faces, "
          f"{vid_acc * 100:.1f}% of {n_sources} videos/images")
    print(f"  Today:   bgr_01, out[0] <= 0.5 means fake: {current * 100:.1f}% of faces")
    if best_acc < 0.6:
        verdict = "no_clear_winner"
        print("  No combination is clearly better than guessing. Do not change app.py yet;")
        print("  check the test files, or the model may need retraining (roadmap Phase 8).")
    elif best_acc - current <= 0.02:
        verdict = "keep_current"
        print("  app.py's current preprocessing is as good as the best one. Keep it;")
        print("  T2.1 only needs to tidy the code, not change verdicts.")
    else:
        verdict = "switch"
        print(f"  app.py should switch to the best combination (+{(best_acc - current) * 100:.1f} points, roadmap T2.1).")

    # Tune the cut-off for the preprocessing app.py uses (bgr_01, low output = fake).
    print("\nFor app.py's preprocessing (bgr_01, output <= cut-off means fake):", end="")
    sweep = threshold_sweep(outputs["bgr_01"], is_fake, sources, 0, False)

    report = {
        "model": model_info,
        "faces": {"real": len(real), "fake": len(fake)},
        "accuracy": [{"mode": m, "reading": r, "face_accuracy": round(a, 4)} for a, m, r in table],
        "best": {"mode": best_mode, "reading": best_reading, "face_accuracy": round(best_acc, 4),
                 "video_accuracy": round(vid_acc, 4), "videos": n_sources},
        "current_app_accuracy": round(current, 4),
        "verdict": verdict,
        "threshold": sweep,
        "mean_outputs": {m: {"real": o[~is_fake].mean(axis=0).round(4).tolist(),
                             "fake": o[is_fake].mean(axis=0).round(4).tolist()} for m, o in outputs.items()},
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {args.report}. Please paste everything from 'Model:' down, or send the JSON file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
