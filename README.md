# Smart Alarm Clock — ESP32-CAM + TinyML + MAX30102

An offline smart-alarm prototype built around the **AI-Thinker ESP32-CAM**. The system uses an onboard camera and a pretrained TensorFlow Lite Micro model to confirm that a person is present, then measures heart rate with a MAX30102 and displays the process on an SSD1306 OLED.

The current demo uses a jumper wire as a manual alarm trigger. MPU6050 activity detection is represented by a clearly labelled simulation because the sensor is not currently connected.

## Current workflow

1. **Wait for trigger:** The OLED displays that the system is ready.
2. **Start alarm:** Briefly connect GPIO13 to GND using a jumper wire.
3. **Person detection:** The camera captures grayscale frames and runs the pretrained 96×96 person/not-person model locally.
4. **Stop alarm:** Person presence is confirmed when the person score is at least 0.65 and is higher than the no-person score for three consecutive frames.
5. **Heart-rate measurement:** The MAX30102 takes one measurement window. The OLED displays the live IR sensor reading, BPM estimate/sample count, and remaining measurement time.
6. **Activity stage:** A 10-second activity stage is simulated. The OLED and Serial Monitor explicitly state that MPU6050 data is not being read.
7. **Complete:** The OLED displays the accepted heart-rate estimate and routine-complete status. The system returns to the trigger-waiting state.

## Hardware

- AI-Thinker ESP32-CAM with OV3660 camera and PSRAM
- MAX30102 pulse sensor module
- 128×64 SSD1306 I2C OLED
- Jumper wire for the manual start trigger
- Optional active buzzer for the alarm output
- MPU6050 is planned but **not connected / not currently read**

### Pin connections

| Component | Module pin | ESP32-CAM |
|---|---|---|
| OLED | SDA | GPIO14 |
| OLED | SCL | GPIO15 |
| OLED | VCC | 3.3V |
| OLED | GND | GND |
| MAX30102 | SDA | GPIO14 |
| MAX30102 | SCL | GPIO15 |
| MAX30102 | VIN | 3.3V (for the tested breakout) |
| MAX30102 | GND | GND |
| MAX30102 | INT | Not connected |
| Start trigger | Jumper wire | GPIO13 to GND, momentarily |
| Buzzer control | Positive/control side | GPIO2 |
| Buzzer | Negative side | GND |

The OLED and MAX30102 share the I2C bus. The tested I2C addresses are **0x3C** (OLED) and **0x57** (MAX30102).

**Wiring cautions**
- All modules must share GND.
- GPIO13 is configured as `INPUT_PULLUP`; momentarily short it to GND to trigger the demo. Remove the jumper after triggering and do not hold it during reset.
- GPIO2 is an ESP32 boot-strapping pin. The buzzer output is configured as active-HIGH in code; avoid a buzzer connection that pulls GPIO2 HIGH during reset or prevents boot. Direct GPIO drive is appropriate only if the buzzer is a 3.3V active buzzer whose current is within the ESP32 GPIO limit. Otherwise use a transistor driver.
- GPIO14 and GPIO15 are also associated with the microSD interface. Do not use the microSD slot while these pins are used for I2C.
- The camera uses the AI-Thinker board's onboard camera connector; no separate camera wiring is required.

## Buzzer public mode

In `src/main.cpp`:

```cpp
#define PUBLIC_MODE true
```

- `true`: silent/public demonstration mode; buzzer output remains LOW.
- `false`: normal alarm mode; GPIO2 is toggled while the alarm is active.

The rest of the alarm workflow runs in either mode.

## Heart-rate measurement behavior

The current demonstration implementation:
- Uses an IR threshold of 50,000 for finger-signal detection (based on the tested module's observed readings).
- Collects one 15-second measurement window.
- Accepts beat-derived BPM samples between 40 and 180 BPM.
- Requires at least three valid BPM samples.
- Restarts the measurement window if finger contact is lost for more than 1.5 seconds.
- Displays the live IR reading and BPM estimate on the OLED during measurement.

These are prototype software thresholds, not clinical criteria. MAX30102 readings are estimates and this project is not a medical device. Keep the finger still and ensure stable contact for better readings.

## Person-detection model

- Pretrained TensorFlow Lite Micro 96×96 grayscale person/not-person classifier.
- Runs locally on the ESP32-CAM; no cloud inference or Wi-Fi connection is required.
- Camera frame: 160×120 grayscale (QQVGA).
- The central square is cropped and resized to 96×96 for inference.
- Person output index: 1; no-person output index: 0.
- Person threshold: 0.65.
- Three consecutive positive frames are required for confirmation.
- The model detects **person presence only**. It does not identify standing, sitting, walking, or posture.

The original working person-detection implementation is available in the repository history / earlier project version. The integrated sketch retains its camera configuration, preprocessing, model initialization, and inference logic.

## Software and dependencies

- Visual Studio Code with PlatformIO
- PlatformIO Espressif32 platform
- Arduino framework for ESP32
- `espressif/esp32-camera`
- `spaziochirale/Chirale_TensorFLowLite`
- Adafruit GFX Library
- Adafruit SSD1306
- SparkFun MAX3010x Pulse and Proximity Sensor Library

The dependency declarations are in `platformio.ini`. PlatformIO resolves compatible package versions during the build.

## Project structure

```text
project/
├── platformio.ini
├── README.md
├── scripts/
│   └── get_model.py
└── src/
    ├── main.cpp
    ├── person_detect_model_data.cpp
    └── person_detect_model_data.h
```

Keep the generated model data files in `src/`. Do not replace them with unrelated model data.

If the model data source needs to be regenerated, run the repository's model download/generation script from the project root:

```bash
python3 scripts/get_model.py
```

## Build, upload, and monitor

From the project root:

**Build**

```bash
~/.platformio/penv/bin/platformio run
```

**Upload** (replace the serial port if yours differs):

```bash
~/.platformio/penv/bin/platformio run --target upload --upload-port /dev/cu.usbserial-A5069RR4
```

**Serial Monitor**

```bash
~/.platformio/penv/bin/platformio device monitor -p /dev/cu.usbserial-A5069RR4 -b 115200
```

For a manual USB-to-TTL adapter, GPIO0 must be connected to GND only while entering flashing mode. Remove GPIO0 from GND and reset the board to run normally. The ESP32-CAM-MB programmer may handle this automatically.

## Verified prototype behavior

The integrated prototype has been run on the physical AI-Thinker ESP32-CAM with OV3660. A successful test log showed:
- OLED initialized at 0x3C.
- MAX30102 initialized successfully.
- Camera sensor PID 0x3660.
- TensorFlow Lite model and tensors initialized.
- GPIO13 trigger received.
- Person confirmed after three consecutive positive frames.
- Alarm-stop transition executed.
- One MAX30102 measurement completed and an accepted estimate was printed.
- Simulated activity stage completed.
- System returned to `WAITING_FOR_TRIGGER`.

Example output from one test included an accepted heart-rate estimate of 74.1 BPM. This is a single prototype run, not an accuracy validation.

## Current limitations / future work

- **MPU6050:** Not integrated for real motion sensing yet; activity is simulated for the demo.
- **RTC:** No RTC module is installed. The alarm is started manually with GPIO13 rather than at a scheduled time.
- **Buzzer:** Public mode is enabled by default. Normal buzzer operation depends on a safe, suitable GPIO2 driver/wiring arrangement.
- **Heart-rate accuracy:** The current simple beat-interval averaging is for demonstration and needs repeatability testing and comparison against a reliable reference.
- **Person detection:** Detects person presence, not posture or wakefulness.
- **Power:** Use a stable supply suitable for the ESP32-CAM, especially while the camera and inference model are active.

This is an educational IoT prototype, not a medical or safety-critical device.
