# Data-Preprocessing-Formative

# User Identity + Product Recommendation System
Formative 2 - Data Preprocessing (Group of 3: Hasbiyallah, Favor, Prince)

This project builds a simulated "smart store" system: a user is verified by
face, gets a product prediction, then confirms with their voice, before the
predicted product is displayed. If either check fails, access is denied.

```
Start -> Facial Recognition --fail--> Access Denied
   |
  pass
   v
Run Product Recommendation Model
   v
Voice Validation --fail--> Access Denied
   |
  pass
   v
Display Predicted Product
```

## Status: complete, using real data

- Real customer social profiles + transaction data (merged, cleaned, validated)
- Real face photos (3 expressions x 3 team members)
- Real voice recordings (2 phrases x 3 team members)
- Real "unauthorized" test files (photo + voice of someone outside the team)
- All 3 models trained and evaluated on real data
- Full CLI simulation tested with both authorized and unauthorized attempts

## Project structure

```
data/
  raw/                       Real exported CSVs (social profiles + transactions)
  images/                    Real team photos: name_expression.jpg
  audio/                     Real team recordings: name_phrase.wav
  unauthorized/              Real impostor test files (NOT used for training)
  processed/
    merged_dataset.csv       Output of the tabular merge + feature engineering
    eda_plots/               4 labeled EDA plots
  image_features.csv         Extracted image features (all augmentations)
  audio_features.csv         Extracted audio features (all augmentations)

scripts/
  merge_data.py                    Task 1: merge + clean + EDA
  image_pipeline.py                Task 2: image collection, augment, extract features
  audio_pipeline.py                Task 3: audio collection, augment, extract features
  train_recommendation_model.py    Task 4: product recommendation model
  train_face_model.py              Task 4: facial recognition model
  train_voice_model.py             Task 4: voiceprint verification model

app/
  cli_simulation.py          Task 5/6: wires all 3 models into one CLI flow

models/                       Trained model .pkl files
notebooks/
  01_data_merge_eda.ipynb     Full pipeline, executed, all outputs + audio playback
```

## How to run it

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 scripts/merge_data.py
python3 scripts/image_pipeline.py
python3 scripts/audio_pipeline.py
python3 scripts/train_recommendation_model.py
python3 scripts/train_face_model.py
python3 scripts/train_voice_model.py

python3 app/cli_simulation.py
```

When prompted by the CLI app, enter a real file path, e.g.:
```
data/images/hasby_neutral.jpg
data/audio/hasby_approve.wav
```
or use the unauthorized test files to see it correctly deny access:
```
data/unauthorized/unauthorized_neutral.jpg
data/unauthorized/unauthorized_confirm.wav
```

## Results (real data)

| Model | Accuracy | F1 (macro) | Log Loss |
|---|---|---|---|
| Facial Recognition | 1.000 | 1.000 | 0.183 |
| Voiceprint Verification | 0.625 | 0.611 | 0.884 |
| Product Recommendation | 0.100 | 0.109 | 1.952 |

Face recognition is near-perfect since 3 real faces are visually distinct, even
with a small dataset. Voice accuracy is lower and expected - real speech has
much more natural variation than a photo, and the dataset is small (24 rows
across 3 people). The recommendation model's low accuracy reflects that the
underlying tabular data (as provided) has no strong learnable signal linking
social/transaction features to product category - an honest finding, not a bug.

**Live simulation example:**
- Authorized attempt (real photo + voice): face confidence 0.97, voice
  confidence 0.89 -> **ACCESS GRANTED**
- Unauthorized attempt (real impostor photo + voice): face confidence dropped
  to 0.41, below the 0.5 threshold -> **ACCESS DENIED** at the first checkpoint

## Team & task split

| Person | Owns |
|---|---|
| Favor | Task 1 (data merge/EDA), Task 4c (recommendation model), final CLI integration, repo management |
| Hasbiyallah | Task 2 (image pipeline), Task 4a (facial recognition model) |
| Prince | Task 3 (audio pipeline), Task 4b (voiceprint verification model) |

Report and demo video are split by section (each person covers the part they built).

## Known limitations

With a small dataset (3 people, few samples each), the voice and recommendation
models show real but modest performance - this is expected and explained above,
not a bug. In rare cases with placeholder/edge-case testing, a plain classifier
can misclassify an unrecognized face/voice, since it always picks one of its
known classes rather than truly detecting "unknown." The final simulation on
real data (above) demonstrates correct behavior on both the authorized and
unauthorized test cases.
