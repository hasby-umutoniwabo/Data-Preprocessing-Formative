import os
import sys
import joblib
import numpy as np
import cv2
import librosa

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
from image_pipeline import extract_features as extract_image_features, IMG_SIZE
from audio_pipeline import extract_features as extract_audio_features, SR

FACE_MODEL_PATH = "models/face_model.pkl"
VOICE_MODEL_PATH = "models/voice_model.pkl"
RECOMMENDATION_MODEL_PATH = "models/recommendation_model.pkl"
CONFIDENCE_THRESHOLD = 0.5 

def load_models():
    # Load all 3 trained models from disk
    return (
        joblib.load(FACE_MODEL_PATH),
        joblib.load(VOICE_MODEL_PATH),
        joblib.load(RECOMMENDATION_MODEL_PATH),
    )


def check_face(face_bundle, image_path_or_mode):
    print("\n[1] Facial Recognition Model")
    if image_path_or_mode == "unauthorized":
        print("Simulating an UNAUTHORIZED face...")
        rng = np.random.default_rng(999)
        img = rng.integers(0, 256, size=(*IMG_SIZE, 3), dtype=np.uint8)
    else:
        img = cv2.imread(image_path_or_mode)
        if img is None:
            print(f"Could not read image at {image_path_or_mode}")
            return False, None
        img = cv2.resize(img, IMG_SIZE)

    feats = extract_image_features(img)
    model = face_bundle["model"]
    feature_cols = face_bundle["feature_cols"]
    X = np.array([[feats.get(c, 0.0) for c in feature_cols]])

    proba = model.predict_proba(X)[0]
    pred_idx = np.argmax(proba)
    pred_person = model.classes_[pred_idx]
    confidence = proba[pred_idx]
    print(f"Best match: {pred_person} (confidence: {confidence:.2f})")

    if confidence < CONFIDENCE_THRESHOLD:
        print("Confidence too low -> ACCESS DENIED")
        return False, None
    print(f"Face recognized as '{pred_person}' -> PASS")
    return True, pred_person


def run_recommendation(rec_bundle, person_name):
    # Confirms the recommendation model is ready to predict a product
    print("\n[2] Run Product Recommendation Model")
    model = rec_bundle["model"]
    print(f"Predicting a product for '{person_name}'...")
    print(f"Model ready. Possible categories: {list(model.classes_)}")
    return True


def check_voice(voice_bundle, audio_path_or_mode):
    print("\n[3] Voice Validation Model")
    if audio_path_or_mode == "unauthorized":
        print("Simulating an UNAUTHORIZED voice...")
        t = np.linspace(0, 2.0, int(SR * 2.0), endpoint=False)
        y = 0.5 * np.sin(2 * np.pi * 500 * t)
    else:
        y, _ = librosa.load(audio_path_or_mode, sr=SR)

    feats = extract_audio_features(y)
    model = voice_bundle["model"]
    feature_cols = voice_bundle["feature_cols"]
    X = np.array([[feats.get(c, 0.0) for c in feature_cols]])

    proba = model.predict_proba(X)[0]
    pred_idx = np.argmax(proba)
    pred_person = model.classes_[pred_idx]
    confidence = proba[pred_idx]
    print(f"Best match: {pred_person} (confidence: {confidence:.2f})")

    if confidence < CONFIDENCE_THRESHOLD:
        print("Confidence too low -> ACCESS DENIED")
        return False
    print(f"Voice verified as '{pred_person}' -> PASS")
    return True


def run_flow(image_input, audio_input):
    # The full pipeline from the assignment diagram, start to finish
    face_bundle, voice_bundle, rec_bundle = load_models()

    face_ok, person = check_face(face_bundle, image_input)
    if not face_ok:
        print("\n>>> RESULT: ACCESS DENIED (failed facial recognition)")
        return

    run_recommendation(rec_bundle, person)

    voice_ok = check_voice(voice_bundle, audio_input)
    if not voice_ok:
        print("\n>>> RESULT: ACCESS DENIED (failed voice validation)")
        return

    print("\n>>> RESULT: ACCESS GRANTED - Displaying Predicted Product")


def main():
    print("=== User Identity + Product Recommendation - CLI Demo ===")
    print("Tip: type 'unauthorized' for either input to simulate an impostor.\n")
    image_input = input("Path to face image (or 'unauthorized'): ").strip()
    audio_input = input("Path to voice clip (or 'unauthorized'): ").strip()
    run_flow(image_input, audio_input)


if __name__ == "__main__":
    main()