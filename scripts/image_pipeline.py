import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IMAGES_DIR = "data/images"
OUTPUT_CSV = "data/image_features.csv"
SAMPLE_PLOT = "data/images/_sample_preview.png"
EXPRESSIONS = ["neutral", "smile", "surprised"]
IMG_SIZE = (128, 128)
PLACEHOLDER_MEMBERS = ["member1", "member2", "member3"]  # replace with real names once photos exist


def find_real_images():
    # Look for files like name_expression.jpg in data/images/
    found = {}
    if not os.path.isdir(IMAGES_DIR):
        return found
    for fname in os.listdir(IMAGES_DIR):
        lower = fname.lower()
        if not lower.endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(lower)[0]
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        expression = parts[-1]
        name = "_".join(parts[:-1])
        if expression in EXPRESSIONS:
            found[(name, expression)] = os.path.join(IMAGES_DIR, fname)
    return found


def make_placeholder_face(expression, seed):
    # Draws a simple cartoon face so the pipeline has something to run on for now
    rng = np.random.default_rng(seed)
    img = np.full((*IMG_SIZE, 3), 255, dtype=np.uint8)
    skin_color = tuple(int(c) for c in rng.integers(150, 230, size=3))
    cx, cy = IMG_SIZE[1] // 2, IMG_SIZE[0] // 2

    cv2.circle(img, (cx, cy), 50, skin_color, -1)
    cv2.circle(img, (cx - 18, cy - 10), 6, (30, 30, 30), -1)
    cv2.circle(img, (cx + 18, cy - 10), 6, (30, 30, 30), -1)

    if expression == "neutral":
        cv2.line(img, (cx - 15, cy + 20), (cx + 15, cy + 20), (30, 30, 30), 2)
    elif expression == "smile":
        cv2.ellipse(img, (cx, cy + 15), (18, 10), 0, 0, 180, (30, 30, 30), 2)
    elif expression == "surprised":
        cv2.circle(img, (cx, cy + 20), 8, (30, 30, 30), 2)
        cv2.line(img, (cx - 25, cy - 25), (cx - 8, cy - 30), (30, 30, 30), 2)
        cv2.line(img, (cx + 8, cy - 30), (cx + 25, cy - 25), (30, 30, 30), 2)
    return img


def load_dataset():
    # Real photos win if they exist, otherwise fall back to placeholders
    dataset = {}
    real_images = find_real_images()
    if real_images:
        print(f"Found {len(real_images)} real image(s) in {IMAGES_DIR}/")
        for (name, expr), path in real_images.items():
            img = cv2.imread(path)
            img = cv2.resize(img, IMG_SIZE)
            dataset[(name, expr)] = img
    else:
        print(f"No real photos found -> generating placeholders. Drop real files into {IMAGES_DIR}/ and re-run.")
        seed = 0
        for name in PLACEHOLDER_MEMBERS:
            for expr in EXPRESSIONS:
                dataset[(name, expr)] = make_placeholder_face(expr, seed)
                seed += 1
    return dataset


def preview_samples(dataset):
    # Save a grid image so we can eyeball that everything loaded correctly
    keys = list(dataset.keys())
    n = min(len(keys), 9)
    cols = 3
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols * 2.2, rows * 2.4))
    for i in range(n):
        name, expr = keys[i]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(cv2.cvtColor(dataset[keys[i]], cv2.COLOR_BGR2RGB))
        plt.title(f"{name}\n{expr}", fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(SAMPLE_PLOT), exist_ok=True)
    plt.savefig(SAMPLE_PLOT, dpi=110)
    print(f"Saved preview grid -> {SAMPLE_PLOT}")


def augment_rotate(img, angle=15):
    # Rotate the image slightly
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderValue=(255, 255, 255))


def augment_flip(img):
    # Mirror the image horizontally
    return cv2.flip(img, 1)


def augment_grayscale(img):
    # Convert to grayscale (kept as 3 channels for consistency)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


AUGMENTATIONS = {
    "original": lambda img: img,
    "rotated": augment_rotate,
    "flipped": augment_flip,
    "grayscale": augment_grayscale,
}


def extract_features(img):
    # Color + grayscale histograms, brightness stats, and a small flattened embedding
    features = {}
    for i, channel in enumerate(["b", "g", "r"]):
        hist = cv2.calcHist([img], [i], None, [16], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-8)
        for j, val in enumerate(hist):
            features[f"hist_{channel}_{j}"] = float(val)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
    gray_hist = gray_hist / (gray_hist.sum() + 1e-8)
    for j, val in enumerate(gray_hist):
        features[f"hist_gray_{j}"] = float(val)

    features["brightness_mean"] = float(gray.mean())
    features["brightness_std"] = float(gray.std())

    small = cv2.resize(gray, (16, 16)).flatten() / 255.0
    for j, val in enumerate(small):
        features[f"embed_{j}"] = float(val)
    return features


def main():
    dataset = load_dataset()
    preview_samples(dataset)

    rows = []
    for (name, expression), img in dataset.items():
        for aug_name, aug_fn in AUGMENTATIONS.items():
            feats = extract_features(aug_fn(img))
            row = {"person": name, "expression": expression, "augmentation": aug_name}
            row.update(feats)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} feature rows ({df.shape[1]} columns) -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()