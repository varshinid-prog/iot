#!/usr/bin/env python3
"""
Fetch the official TensorFlow Lite Micro int8 grayscale person-detection model.
No training is performed. This script extracts the model C source from Google's
published archive and normalizes it to this project's symbol names.
"""
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import io
import re
import sys

URL = "https://storage.googleapis.com/download.tensorflow.org/data/tf_lite_micro_person_data_int8_grayscale_2020_01_13.zip"
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

print("Downloading official pretrained int8 grayscale person model...")
try:
    data = urlopen(URL, timeout=60).read()
except Exception as exc:
    print(f"Download failed: {exc}")
    print("Try again on a stable internet connection.")
    sys.exit(1)

print(f"Downloaded {len(data):,} bytes")
try:
    archive = ZipFile(io.BytesIO(data))
except Exception as exc:
    print(f"Archive could not be opened: {exc}")
    sys.exit(1)

names = archive.namelist()
candidates = [n for n in names if n.lower().endswith((".cc", ".cpp")) and "person_detect_model_data" in n.lower()]
if not candidates:
    candidates = [n for n in names if n.lower().endswith((".cc", ".cpp")) and "person" in n.lower() and "model" in n.lower()]

if candidates:
    raw = archive.read(candidates[0]).decode("utf-8", errors="replace")
    # Keep the upstream generated model array, but make it a normal C++ source.
    # Rename symbols so they match the project's header.
    raw = re.sub(r"\b g_person_detect_model_data_len\b", " g_person_detect_model_data_len", raw)
    # Most upstream generated files already use these symbol names.
    if "g_person_detect_model_data" not in raw:
        print("Found C source, but expected model symbol was not present.")
        print("Archive members:", *names, sep="\n  ")
        sys.exit(2)
    # Add a stable length symbol if the upstream file omitted it.
    if "g_person_detect_model_data_len" not in raw:
        raw += "\nconst int g_person_detect_model_data_len = sizeof(g_person_detect_model_data);\n"
    (SRC / "person_detect_model_data.cpp").write_text(raw)
    print(f"Installed model source from: {candidates[0]}")
else:
    print("No generated C/C++ model file found in archive.")
    print("Archive members:", *names, sep="\n  ")
    sys.exit(2)

print("Model source installed. Reopen/build the PlatformIO project.")
