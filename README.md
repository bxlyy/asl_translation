# ASL-to-English Stacked Translation Pipeline

This repository implements a stacked neural architecture for real-time American Sign Language (ASL) to English translation, inspired by the cascaded pipeline described in [Computers 15-00020-v2](references/computers-15-00020-v2.pdf).

The translation pipeline is split into two distinct stages:
1.  **Computer Vision (Model 1):** Video frames (MediaPipe landmarks) &rarr; ASL Gloss (CTC Loss).
2.  **Machine Translation (Model 2):** ASL Gloss &rarr; English Translation (Transformer/Seq2Seq).

This architecture allows for high interpretability and makes it lightweight enough to run in real-time as a mobile app (iOS) connected to a backend.

---

## Directory Structure

```text
├── data_prep/
│   ├── 1_train_english_to_gloss.ipynb   # Trains the NLLB Seq2Seq model on ASLG-PC12
│   ├── 2_generate_how2sign_gloss.ipynb  # Translates How2Sign English sentences to ASL Gloss
│   └── 3_extract_how2sign_features.ipynb # Extracts MediaPipe landmarks & motion velocity
├── models/
│   └── train_keypoint_to_gloss.ipynb    # Trains the 1D-CNN + BiLSTM CTC model
├── references/
│   └── computers-15-00020-v2.pdf        # Baseline architecture research paper
├── deprecated/                          # WLASL isolated sign files (archived)
│   ├── train_wlasl_isolated.ipynb
│   └── WLASL_Dataset/
├── final_asl_gloss_model/               # Saved weights for the English -> Gloss NLLB model
├── how2sign_dataset/                    # Raw clips and aligned translation metadata
└── how2sign_features/                   # Precomputed landmark sequences (.npy)
```

---

## Step-by-Step Dataset & Model Replication

To rebuild the training datasets and train the translation models from scratch, run the notebooks in the following order:

### Phase 1: Text Translator (English &rarr; Gloss)
We use a Seq2Seq transformer to translate English sentences to ASL glosses to generate ground-truth labels for our vision dataset.
1.  **Train the Translator:** Run `data_prep/1_train_english_to_gloss.ipynb`. This loads the `ASLG-PC12_corpus.csv` dataset and fine-tunes `facebook/nllb-200-distilled-600M` to translate standard English to Pidgin-style ASL gloss syntax. The final weights are saved to `final_asl_gloss_model/`.
2.  **Generate How2Sign Labels:** Run `data_prep/2_generate_how2sign_gloss.ipynb`. This loads the How2Sign text translation files (e.g., `how2sign_realigned_test.csv`), passes the English sentences through the fine-tuned NLLB model, and creates a new metadata file containing the aligned ASL Glosses (e.g., `how2sign_realigned_test_glosses.csv`).

### Phase 2: Feature Extraction (Videos &rarr; Coordinates)
3.  **MediaPipe Feature Processing:** Run `data_prep/3_extract_how2sign_features.ipynb`. This loops through the raw video clips (`.mp4`), extracts spatial keypoints (Pose, Face, and Hands) using MediaPipe Holistic, calculates the temporal velocity delta, and saves the sequence as normalized `.npy` matrices of shape `(num_frames, 510)` in `how2sign_features/`.

### Phase 3: CV Model (Keypoints &rarr; Gloss)
4.  **Train the Vision Model:** Run `models/train_keypoint_to_gloss.ipynb`. This PyTorch notebook loads the precomputed feature matrices and target glosses, processes them through a **1D-CNN + BiLSTM**, and trains them using **CTC (Connectionist Temporal Classification) Loss** on your GPU (supporting Apple Metal Performance Shaders - MPS).

---

## Deployment & Feasibility Design

To host this translation model as a real-time service in a mobile app:
1.  **Frontend (iOS):** The iPhone app uses the camera to run Google MediaPipe Holistic locally on the device (highly optimized, runs at 30-60 FPS).
2.  **Network Payload:** Instead of uploading heavy video streams, the app packages the `(x, y, z)` spatial coordinates (a tiny 2KB payload per frame) and streams them to the server via WebSockets.
3.  **Backend (FastAPI):** A lightweight Python backend receives the keypoints, calculates velocity, passes the 510-dim vectors through the 1D-CNN + BiLSTM model (Model 1) to predict the ASL Gloss sequence, and translates the gloss to English using the NLLB model (Model 2). The translated text is returned to the phone in real-time.
