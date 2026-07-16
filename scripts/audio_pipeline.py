"""
Task 3: collect voice recordings, augment them, extract features into audio_features.csv.
Looks for real clips in data/audio/ named like name_phrase.wav
(phrase = approve/confirm). If none exist yet, generates placeholder tones
instead so the rest of the pipeline can be built now.
"""

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

AUDIO_DIR = "data/audio"
OUTPUT_CSV = "data/audio_features.csv"
SAMPLE_PLOT = "data/audio/_sample_preview.png"
PHRASES = ["approve", "confirm"]  # "Yes, approve" / "Confirm transaction"
SR = 16000
DURATION = 2.0
PLACEHOLDER_MEMBERS = ["member1", "member2", "member3"]  # replace with real names once recordings exist


def find_real_audio():
    # Look for files like name_phrase.wav in data/audio/
    found = {}
    if not os.path.isdir(AUDIO_DIR):
        return found
    for fname in os.listdir(AUDIO_DIR):
        if not fname.lower().endswith(".wav"):
            continue
        stem = os.path.splitext(fname.lower())[0]
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        phrase = parts[-1]
        name = "_".join(parts[:-1])
        if phrase in PHRASES:
            found[(name, phrase)] = os.path.join(AUDIO_DIR, fname)
    return found


def make_placeholder_clip(phrase, seed):
    # Generates a short synthetic tone so the pipeline has something to run on for now
    rng = np.random.default_rng(seed)
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    base_freq = rng.uniform(110, 220)
    signal = (
        0.5 * np.sin(2 * np.pi * base_freq * t)
        + 0.25 * np.sin(2 * np.pi * base_freq * 2 * t)
        + 0.15 * np.sin(2 * np.pi * base_freq * 3 * t)
    )
    n_syllables = 2 if phrase == "approve" else 3
    envelope = np.zeros_like(t)
    syll_len = len(t) // (n_syllables * 2)
    for i in range(n_syllables):
        start = i * syll_len * 2
        envelope[start:start + syll_len] = 1.0
    envelope = np.convolve(envelope, np.ones(200) / 200, mode="same")
    signal = signal * envelope + rng.normal(0, 0.01, size=signal.shape)
    signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.8
    return signal.astype(np.float32)


def load_dataset():
    # Real recordings win if they exist, otherwise fall back to placeholders
    dataset = {}
    real_audio = find_real_audio()
    if real_audio:
        print(f"Found {len(real_audio)} real recording(s) in {AUDIO_DIR}/")
        for (name, phrase), path in real_audio.items():
            y, _ = librosa.load(path, sr=SR)
            dataset[(name, phrase)] = y
    else:
        print(f"No real recordings found -> generating placeholders. Drop .wav files into {AUDIO_DIR}/ and re-run.")
        seed = 0
        for name in PLACEHOLDER_MEMBERS:
            for phrase in PHRASES:
                dataset[(name, phrase)] = make_placeholder_clip(phrase, seed)
                seed += 1
    return dataset


def preview_samples(dataset):
    # Save a waveform + spectrogram plot for a few samples so we can eyeball them
    keys = list(dataset.keys())
    n = min(len(keys), 3)
    fig, axes = plt.subplots(n, 2, figsize=(10, 2.6 * n))
    if n == 1:
        axes = [axes]
    for i in range(n):
        name, phrase = keys[i]
        y = dataset[keys[i]]
        librosa.display.waveshow(y, sr=SR, ax=axes[i][0])
        axes[i][0].set_title(f"{name} - {phrase} (waveform)", fontsize=9)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=SR, x_axis="time", y_axis="hz", ax=axes[i][1])
        axes[i][1].set_title(f"{name} - {phrase} (spectrogram)", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(SAMPLE_PLOT), exist_ok=True)
    plt.savefig(SAMPLE_PLOT, dpi=110)
    print(f"Saved waveform + spectrogram preview -> {SAMPLE_PLOT}")


def augment_pitch_shift(y, sr=SR):
    # Shift the pitch up a few semitones
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=3)


def augment_add_noise(y, sr=SR):
    # Add a bit of background noise
    return y + np.random.normal(0, 0.02, size=y.shape)


def augment_time_stretch(y, sr=SR):
    # Speed the clip up slightly, then pad/trim back to the original length
    stretched = librosa.effects.time_stretch(y, rate=1.2)
    if len(stretched) < len(y):
        stretched = np.pad(stretched, (0, len(y) - len(stretched)))
    else:
        stretched = stretched[:len(y)]
    return stretched


AUGMENTATIONS = {
    "original": lambda y: y,
    "pitch_shifted": augment_pitch_shift,
    "noisy": augment_add_noise,
    "time_stretched": augment_time_stretch,
}


def extract_features(y, sr=SR):
    # MFCCs, spectral roll-off, energy (RMS), zero-crossing rate
    features = {}
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(mfccs.shape[0]):
        features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
        features[f"mfcc_{i}_std"] = float(np.std(mfccs[i]))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_rolloff_std"] = float(np.std(rolloff))

    rms = librosa.feature.rms(y=y)
    features["energy_mean"] = float(np.mean(rms))
    features["energy_std"] = float(np.std(rms))

    zcr = librosa.feature.zero_crossing_rate(y=y)
    features["zero_crossing_rate_mean"] = float(np.mean(zcr))
    return features


def main():
    dataset = load_dataset()
    preview_samples(dataset)

    rows = []
    for (name, phrase), y in dataset.items():
        for aug_name, aug_fn in AUGMENTATIONS.items():
            feats = extract_features(aug_fn(y))
            row = {"person": name, "phrase": phrase, "augmentation": aug_name}
            row.update(feats)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} feature rows ({df.shape[1]} columns) -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()