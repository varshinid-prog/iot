# ESP32-CAM pretrained person detection (PlatformIO)

## What this project does
Runs Google's pre-trained TensorFlow Lite Micro 96x96 grayscale person/not-person
classifier locally on an AI-Thinker ESP32-CAM. No training or cloud inference is
required. It prints scores over Serial and lights GPIO4 after three consecutive
person-positive frames.

This is a **person-presence classifier**, not a standing-posture detector.

## Important compatibility note
This package uses the older `TensorFlowLite_ESP32` PlatformIO library as a
practical Arduino/PlatformIO route. Its maintainer marks it as outdated and
recommends Espressif's newer official `esp-tflite-micro` integration. We use it
here only because the official current integration is primarily an ESP-IDF
component and the EIM setup has been problematic. Do not treat this as the
preferred long-term production stack.

The model is Google's published int8 grayscale person-detection model. The
script below fetches the model source; no training is performed.

## 1. Generate/download model source
From the project root in a terminal:

```bash
python3 scripts/get_model.py
```

This creates:
`src/person_detect_model_data.cpp`

The header declaration is already present. If the script reports unexpected
archive contents, send the complete output rather than substituting a random
model header.

## 2. Build
Open the project folder in VS Code with PlatformIO installed.
Wait for library installation, then use PlatformIO: Build.

## 3. Upload
For the AI-Thinker ESP32-CAM:
- Programmer: ESP32-CAM-MB or USB-to-TTL adapter.
- Stable 5V supply.
- For manual USB-to-TTL: GPIO0 to GND only while flashing; remove it after upload.
- Select the detected serial port in PlatformIO.
- Upload, then open Serial Monitor at 115200 baud.
- Reset the board after upload if required.

## 4. Expected serial output
You should see camera and model initialization messages, then lines like:

`person=... no_person=... result=PERSON_PENDING/PERSON/NO_PERSON`

Test with an empty scene and then a person in view, under good lighting. The
threshold is a starting value, not a validated accuracy guarantee.

## Hardware notes
- AI-Thinker camera pin mapping is included.
- OV3660 is listed as supported by Espressif's esp32-camera driver, but actual
  initialization and grayscale capture must be confirmed on your module.
- The onboard flash LED is GPIO4 and may affect image exposure; this sketch
  uses it only as a detection indicator. If it changes exposure, disconnect the
  LED behavior by setting `kFlashLedPin` to -1 and guarding pin writes.
- The original ESP32 has limited RAM. The code tries PSRAM for the tensor arena
  first, then internal RAM. If tensor allocation fails, capture the full log.
- The official TensorFlow Micro model uses `kPersonIndex = 1` and
  `kNotAPersonIndex = 0`. The sketch follows that ordering. If you replace the
  model, verify its labels and output tensor before trusting the result.

## Known limitations / not yet verified
This source has not been compiled against your exact PlatformIO package versions
or flashed on your physical OV3660 board in this environment. A successful
PlatformIO build and a live camera test are still required. The old TFLite ESP32
library may need compatibility adjustments with newer ESP32 Arduino cores.
