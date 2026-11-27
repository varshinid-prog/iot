/*
 * SMART ALARM CLOCK - ESP32-CAM + TinyML + OLED + MAX30102
 * Board: AI-Thinker ESP32-CAM (OV3660)
 *
 * Person-detection code is adapted from the working original in:
 * https://github.com/varshinid-prog/iot/blob/main/src/main.cpp
 *
 * Workflow:
 * 1. Momentarily connect START_TRIGGER_PIN (GPIO13) to GND.
 * 2. Buzzer rings while the camera runs the original person/not-person model.
 * 3. Three consecutive person-positive frames stop the alarm.
 * 4. MAX30102 takes one heart-rate measurement after person detection.
 * 5. MPU6050 monitors real accelerometer and gyroscope movement.
 * 6. OLED displays heart-rate, live movement status and routine completion.
 *
 * Important limitations:
 * - The model detects person presence, NOT standing posture.
 * - MPU6050 movement thresholds are configurable and may need calibration for mounting and user movement.
 * - MAX30102 heart-rate values are demonstration estimates, not medical readings.
 * - No RTC is installed: the jumper is a manual alarm-start trigger for the demo.
 */

#include <Arduino.h>
#include <math.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "MAX30105.h"
#include "heartRate.h"

#include "esp_heap_caps.h"
#include "esp_camera.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "person_detect_model_data.h"

// ------------------------- PIN CONFIGURATION -------------------------
// AI-Thinker ESP32-CAM camera pinout (keep unchanged)
#define CAM_PIN_PWDN     32
#define CAM_PIN_RESET    -1
#define CAM_PIN_XCLK      0
#define CAM_PIN_SIOD     26
#define CAM_PIN_SIOC     27
#define CAM_PIN_Y9       35
#define CAM_PIN_Y8       34
#define CAM_PIN_Y7       39
#define CAM_PIN_Y6       36
#define CAM_PIN_Y5       21
#define CAM_PIN_Y4       19
#define CAM_PIN_Y3       18
#define CAM_PIN_Y2        5
#define CAM_PIN_VSYNC    25
#define CAM_PIN_HREF     23
#define CAM_PIN_PCLK     22

// OLED + MAX30102 share the same I2C bus.
static constexpr int I2C_SDA_PIN = 14;
static constexpr int I2C_SCL_PIN = 15;
static constexpr uint8_t OLED_ADDRESS = 0x3C;

// GPIO13 is the manual start trigger: briefly connect it to GND.
static constexpr int START_TRIGGER_PIN = 13;

// Buzzer mode: true = PUBLIC mode (silent), false = normal alarm sound.
#define PUBLIC_MODE true

// GPIO2 controls the active-HIGH buzzer output.
static constexpr int BUZZER_PIN = 2;

static void setBuzzer(bool on) {
    if (PUBLIC_MODE) {
        digitalWrite(BUZZER_PIN, LOW);
    } else {
        digitalWrite(BUZZER_PIN, on ? HIGH : LOW);
    }
}
static constexpr int kFlashLedPin = 4; // Original onboard flash LED indicator

// ------------------------- MODEL CONFIGURATION -------------------------
static constexpr int kInputWidth = 96;
static constexpr int kInputHeight = 96;
static constexpr size_t kTensorArenaBytes = 240 * 1024;
static uint8_t *tensorArena = nullptr;

static const float kPersonThreshold = 0.65f;
static constexpr int kFramesToConfirm = 3;

static const tflite::Model *model = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *inputTensor = nullptr;
static TfLiteTensor *outputTensor = nullptr;

static int consecutivePersonFrames = 0;
static uint32_t inferenceCount = 0;

// ------------------------- SENSOR / DISPLAY -------------------------
static Adafruit_SSD1306 display(128, 64, &Wire, -1);
static MAX30105 particleSensor;
static bool oledReady = false;
static bool maxReady = false;
static bool mpuReady = false;

static constexpr long kFingerIrThreshold = 50000; // Based on this module's observed readings
static constexpr uint32_t kPulseWindowMs = 15000;
static constexpr uint32_t kFingerLossTimeoutMs = 1500;
static constexpr int kMinimumValidBpmSamples = 3;
// Handheld activity verification: require a sustained, repeatable rhythm.
// The sensor measures movement patterns, not physiological exertion.
static constexpr uint32_t kMpuSampleIntervalMs = 20;       // 50 Hz
static constexpr uint32_t kRequiredMovementMs = 25000;     // 25 seconds
static constexpr uint32_t kActivityReminderMs = 45000;
static constexpr uint32_t kRhythmMinIntervalMs = 350;     // about 2.85 Hz max
static constexpr uint32_t kRhythmMaxIntervalMs = 1200;    // about 0.83 Hz min
static constexpr uint32_t kRhythmGapTimeoutMs = 1600;
static constexpr uint32_t kMinimumIntervalsToStart = 4;   // reject isolated shakes
static constexpr float kLinearPeakThreshold = 0.90f;      // m/s^2
static constexpr float kLinearResetThreshold = 0.35f;     // hysteresis
static constexpr float kGyroToLinearFactor = 0.60f;       // rad/s -> activity score
static constexpr float kGravityFilterAlpha = 0.02f;
static constexpr float kActivityFilterAlpha = 0.25f;
static constexpr uint32_t kDoneScreenMs = 12000;

static bool pulseWindowActive = false;
static uint32_t pulseWindowStartMs = 0;
static uint32_t lastFingerSeenMs = 0;
static uint32_t lastBeatMs = 0;
static long currentIrValue = 0;
static float latestBpm = 0.0f;
static float bpmSum = 0.0f;
static int validBpmSamples = 0;
static float acceptedHeartRateBpm = 0.0f;

// MPU rhythmic activity tracking.
static uint32_t lastMpuSampleMs = 0;
static uint32_t lastValidMpuSampleMs = 0;
static uint32_t lastActivityDisplayMs = 0;
static uint32_t accumulatedMovementMs = 0;
static uint32_t activityStartMs = 0;
static uint32_t lastRhythmPeakMs = 0;
static uint32_t rhythmStartMs = 0;
static uint32_t rhythmIntervalCount = 0;
static float gravityMagnitudeEstimate = 9.80665f;
static float filteredActivitySignal = 0.0f;
static float latestDynamicAcceleration = 0.0f;
static float latestGyroMagnitude = 0.0f;
static bool gravityEstimateReady = false;
static bool activitySignalArmed = true;
static bool rhythmActive = false;
static bool movementDetectedThisSample = false;

// ------------------------- STATE MACHINE -------------------------
enum SystemState {
    WAITING_FOR_TRIGGER,
    ALARM_RINGING,
    MEASURE_HEART_RATE,
    DETECT_ACTIVITY,
    ROUTINE_COMPLETE
};

static SystemState systemState = WAITING_FOR_TRIGGER;
static uint32_t stateStartMs = 0;
static uint32_t lastDisplayUpdateMs = 0;
static uint32_t lastBuzzerToggleMs = 0;
static bool buzzerIsOn = false;
static bool triggerLatched = false;

// ------------------------- OLED HELPERS -------------------------
static void showScreen(const char *line1,
                       const char *line2 = "",
                       const char *line3 = "",
                       const char *line4 = "") {
    if (!oledReady) return;

    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setTextWrap(false);
    display.setCursor(0, 0);
    display.println(line1);
    display.println(line2);
    display.println(line3);
    display.println(line4);
    display.display();
}

static void showPulseScreen(const char *heading) {
    if (!oledReady) return;

    char line2[22];
    char line3[22];
    char line4[22];

    if (!pulseWindowActive) {
        snprintf(line2, sizeof(line2), "Place finger steadily");
        snprintf(line3, sizeof(line3), "IR SENSOR: %ld", currentIrValue);
        snprintf(line4, sizeof(line4), "Waiting for signal");
    } else {
        const uint32_t elapsed = millis() - pulseWindowStartMs;
        const uint32_t remaining = (elapsed >= kPulseWindowMs)
                                       ? 0
                                       : (kPulseWindowMs - elapsed + 999) / 1000;
        if (validBpmSamples > 0) {
            snprintf(line3, sizeof(line3), "BPM: %.0f (%d)", latestBpm, validBpmSamples);
        } else {
            snprintf(line3, sizeof(line3), "BPM: detecting...");
        }
        snprintf(line2, sizeof(line2), "IR SENSOR: %ld", currentIrValue);
        snprintf(line4, sizeof(line4), "Hold still: %lus", (unsigned long)remaining);
    }

    showScreen(heading, line2, line3, line4);
}

static void showActivityScreen() {
    if (!oledReady) return;

    char line2[22];
    char line3[22];
    char line4[22];

    const uint32_t progressSeconds = accumulatedMovementMs / 1000;
    const uint32_t requiredSeconds = kRequiredMovementMs / 1000;

    snprintf(line2, sizeof(line2), "Rhythm: %s",
             rhythmActive ? "ACTIVE" : (movementDetectedThisSample ? "CHECKING" : "WAITING"));
    snprintf(line3, sizeof(line3), "Signal: %.1f", filteredActivitySignal);
    snprintf(line4, sizeof(line4), "Progress: %lu/%lu sec",
             (unsigned long)progressSeconds, (unsigned long)requiredSeconds);

    showScreen("REPEAT RHYTHMIC MOVE", line2, line3, line4);
}

static void enterState(SystemState nextState) {
    systemState = nextState;
    stateStartMs = millis();
    lastDisplayUpdateMs = 0;

    switch (systemState) {
        case WAITING_FOR_TRIGGER:
            Serial.println("[STATE] WAITING_FOR_TRIGGER");
            showScreen("SMART ALARM CLOCK", "Ready for demo", "Short GPIO13 to GND", "to start alarm");
            break;

        case ALARM_RINGING:
            Serial.println("[STATE] ALARM_RINGING - show yourself to camera");
            consecutivePersonFrames = 0;
            buzzerIsOn = false;
            lastBuzzerToggleMs = 0;
            showScreen("ALARM RINGING!", "Look at the camera", "Person detection", "will stop the alarm");
            break;

        case MEASURE_HEART_RATE:
            Serial.println("[STATE] MEASURE_HEART_RATE");
            showPulseScreen("MEASURE HEART RATE");
            break;

        case DETECT_ACTIVITY:
            Serial.println("[STATE] DETECT_ACTIVITY - sustained handheld rhythmic movement verification");
            activityStartMs = millis();
            lastMpuSampleMs = 0;
            lastValidMpuSampleMs = 0;
            lastActivityDisplayMs = 0;
            accumulatedMovementMs = 0;
            lastRhythmPeakMs = 0;
            rhythmStartMs = 0;
            rhythmIntervalCount = 0;
            gravityMagnitudeEstimate = 9.80665f;
            filteredActivitySignal = 0.0f;
            latestDynamicAcceleration = 0.0f;
            latestGyroMagnitude = 0.0f;
            gravityEstimateReady = false;
            activitySignalArmed = true;
            rhythmActive = false;
            movementDetectedThisSample = false;
            showScreen("ACTIVITY DETECTION", "Hold sensor securely", "Repeat rhythmic moves", "for 25 seconds");
            break;

        case ROUTINE_COMPLETE: {
            char line2[22];
            char line3[22];
            snprintf(line2, sizeof(line2), "Heart rate: %.0f BPM", acceptedHeartRateBpm);
            snprintf(line3, sizeof(line3), "Movement confirmed");
            Serial.printf("[DONE] accepted heart rate=%.1f BPM\n", acceptedHeartRateBpm);
            showScreen("WAKE ROUTINE DONE", line2, line3, "Have a good day!");
            break;
        }
    }
}

// ------------------------- ORIGINAL CAMERA INITIALIZATION -------------------------
static bool initCamera() {
    camera_config_t config = {};
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = CAM_PIN_Y2;
    config.pin_d1 = CAM_PIN_Y3;
    config.pin_d2 = CAM_PIN_Y4;
    config.pin_d3 = CAM_PIN_Y5;
    config.pin_d4 = CAM_PIN_Y6;
    config.pin_d5 = CAM_PIN_Y7;
    config.pin_d6 = CAM_PIN_Y8;
    config.pin_d7 = CAM_PIN_Y9;
    config.pin_xclk = CAM_PIN_XCLK;
    config.pin_pclk = CAM_PIN_PCLK;
    config.pin_vsync = CAM_PIN_VSYNC;
    config.pin_href = CAM_PIN_HREF;
    config.pin_sccb_sda = CAM_PIN_SIOD;
    config.pin_sccb_scl = CAM_PIN_SIOC;
    config.pin_pwdn = CAM_PIN_PWDN;
    config.pin_reset = CAM_PIN_RESET;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_GRAYSCALE;
    config.frame_size = FRAMESIZE_QQVGA; // 160x120
    config.jpeg_quality = 12;
    config.fb_count = 1;

    if (psramFound()) {
        config.fb_location = CAMERA_FB_IN_PSRAM;
        config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    } else {
        config.fb_location = CAMERA_FB_IN_DRAM;
        config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("[CAM] init failed: 0x%x\n", err);
        return false;
    }

    sensor_t *sensor = esp_camera_sensor_get();
    if (sensor) {
        Serial.printf("[CAM] sensor PID: 0x%04x\n", sensor->id.PID);
    }
    return true;
}

/*
 * Original model preprocessing: crop the central 120x120 square from the
 * 160x120 grayscale camera frame and resize to 96x96. Preserve the original
 * int8 conversion used by the working repository version.
 */
static bool preprocessFrame(const camera_fb_t *fb) {
    if (!fb || !fb->buf || fb->format != PIXFORMAT_GRAYSCALE) return false;
    if (!inputTensor || !inputTensor->data.raw) return false;
    if (fb->width < fb->height || fb->height < kInputHeight) return false;

    const int cropSize = fb->height;
    const int cropX = (fb->width - cropSize) / 2;

    for (int y = 0; y < kInputHeight; ++y) {
        const int sourceY = (y * cropSize) / kInputHeight;
        for (int x = 0; x < kInputWidth; ++x) {
            const int sourceX = cropX + (x * cropSize) / kInputWidth;
            const uint8_t pixel = fb->buf[sourceY * fb->width + sourceX];
            const int index = y * kInputWidth + x;

            if (inputTensor->type == kTfLiteInt8) {
                inputTensor->data.int8[index] = static_cast<int8_t>(pixel ^ 0x80);
            } else if (inputTensor->type == kTfLiteUInt8) {
                inputTensor->data.uint8[index] = pixel;
            } else {
                Serial.printf("[ML] unsupported input tensor type: %d\n", inputTensor->type);
                return false;
            }
        }
    }
    return true;
}

static bool initModel() {
    model = tflite::GetModel(g_person_detect_model_data);
    if (!model) {
        Serial.println("[ML] model pointer is null");
        return false;
    }

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("[ML] schema mismatch: model=%d runtime=%d\n",
                      model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    static tflite::AllOpsResolver resolver;

    tensorArena = static_cast<uint8_t *>(
        heap_caps_malloc(kTensorArenaBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (!tensorArena) {
        tensorArena = static_cast<uint8_t *>(
            heap_caps_malloc(kTensorArenaBytes, MALLOC_CAP_8BIT));
    }
    if (!tensorArena) {
        Serial.println("[ML] tensor arena allocation failed");
        return false;
    }

    static tflite::MicroInterpreter staticInterpreter(
        model, resolver, tensorArena, kTensorArenaBytes);
    interpreter = &staticInterpreter;

    TfLiteStatus allocStatus = interpreter->AllocateTensors();
    if (allocStatus != kTfLiteOk) {
        Serial.println("[ML] AllocateTensors failed");
        return false;
    }

    inputTensor = interpreter->input(0);
    outputTensor = interpreter->output(0);
    if (!inputTensor || !outputTensor) {
        Serial.println("[ML] input/output tensor missing");
        return false;
    }

    Serial.printf("[ML] input: %d x %d x %d, type=%d, scale=%f, zero=%d\n",
                  inputTensor->dims->data[1], inputTensor->dims->data[2],
                  inputTensor->dims->data[3], inputTensor->type,
                  inputTensor->params.scale, inputTensor->params.zero_point);
    Serial.printf("[ML] output type=%d elements=%d\n",
                  outputTensor->type, outputTensor->dims->data[1]);

    if (inputTensor->dims->data[1] != kInputHeight ||
        inputTensor->dims->data[2] != kInputWidth ||
        inputTensor->dims->data[3] != 1) {
        Serial.println("[ML] unexpected model input dimensions");
        return false;
    }
    return true;
}

static float outputScore(int index) {
    if (!outputTensor || index >= outputTensor->dims->data[1]) return 0.0f;

    if (outputTensor->type == kTfLiteInt8) {
        return (static_cast<int32_t>(outputTensor->data.int8[index]) -
                outputTensor->params.zero_point) * outputTensor->params.scale;
    }
    if (outputTensor->type == kTfLiteUInt8) {
        return (static_cast<int32_t>(outputTensor->data.uint8[index]) -
                outputTensor->params.zero_point) * outputTensor->params.scale;
    }
    if (outputTensor->type == kTfLiteFloat32) {
        return outputTensor->data.f[index];
    }
    return 0.0f;
}

// Returns true when an inference completed; writes confirmed person state.
static bool runPersonDetection(bool &confirmedPerson) {
    confirmedPerson = false;
    const uint32_t startMs = millis();

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("[CAM] frame capture failed");
        return false;
    }

    const bool preprocessed = preprocessFrame(fb);
    esp_camera_fb_return(fb);
    if (!preprocessed) {
        Serial.println("[ML] preprocessing failed");
        return false;
    }

    const uint32_t invokeStart = millis();
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("[ML] Invoke failed");
        digitalWrite(kFlashLedPin, LOW);
        return false;
    }

    // Official person-detection model output order: 1 = person, 0 = not-person.
    const float personScore = outputScore(1);
    const float noPersonScore = outputScore(0);
    const bool personThisFrame = personScore >= kPersonThreshold &&
                                 personScore > noPersonScore;

    consecutivePersonFrames = personThisFrame ? consecutivePersonFrames + 1 : 0;
    confirmedPerson = consecutivePersonFrames >= kFramesToConfirm;
    digitalWrite(kFlashLedPin, confirmedPerson ? HIGH : LOW);

    ++inferenceCount;
    Serial.printf("[%lu] person=%.3f no_person=%.3f result=%s inference=%lu ms total=%lu ms\n",
                  static_cast<unsigned long>(inferenceCount), personScore, noPersonScore,
                  confirmedPerson ? "PERSON" : (personThisFrame ? "PERSON_PENDING" : "NO_PERSON"),
                  static_cast<unsigned long>(millis() - invokeStart),
                  static_cast<unsigned long>(millis() - startMs));
    return true;
}

// ------------------------- MAX30102 HEART-RATE DEMO -------------------------
static void drainPulseFifo() {
    particleSensor.check();
    while (particleSensor.available()) {
        particleSensor.nextSample();
    }
}

static void resetPulseWindow() {
    pulseWindowActive = false;
    pulseWindowStartMs = 0;
    lastFingerSeenMs = 0;
    lastBeatMs = 0;
    latestBpm = 0.0f;
    bpmSum = 0.0f;
    validBpmSamples = 0;
}

static void startPulseMeasurement() {
    resetPulseWindow();
    currentIrValue = 0;
    drainPulseFifo();
}

static bool pollPulseSensor() {
    particleSensor.check();

    while (particleSensor.available()) {
        currentIrValue = particleSensor.getIR();
        const uint32_t now = millis();

        if (currentIrValue >= kFingerIrThreshold) {
            lastFingerSeenMs = now;

            if (!pulseWindowActive) {
                pulseWindowActive = true;
                pulseWindowStartMs = now;
                lastBeatMs = 0;
                bpmSum = 0.0f;
                validBpmSamples = 0;
                latestBpm = 0.0f;
                Serial.println("[PULSE] Finger signal detected; measurement window started");
            }

            if (checkForBeat(currentIrValue)) {
                if (lastBeatMs != 0) {
                    const uint32_t delta = now - lastBeatMs;
                    if (delta > 0) {
                        const float bpm = 60000.0f / static_cast<float>(delta);
                        if (bpm >= 40.0f && bpm <= 180.0f) {
                            latestBpm = bpm;
                            bpmSum += bpm;
                            ++validBpmSamples;
                            Serial.printf("[PULSE] beat %.1f BPM (valid samples: %d)\n",
                                          bpm, validBpmSamples);
                        }
                    }
                }
                lastBeatMs = now;
            }
        }

        particleSensor.nextSample();
    }

    // If finger contact is lost, discard the partial window and ask for a steady retry.
    if (pulseWindowActive && millis() - lastFingerSeenMs > kFingerLossTimeoutMs) {
        Serial.println("[PULSE] Finger contact lost; restarting measurement window");
        resetPulseWindow();
    }

    return pulseWindowActive &&
           (millis() - pulseWindowStartMs >= kPulseWindowMs) &&
           (validBpmSamples >= kMinimumValidBpmSamples);
}

static void beginHeartRateMeasurement() {
    startPulseMeasurement();
    enterState(MEASURE_HEART_RATE);
}

static void updatePulseUi() {
    if (millis() - lastDisplayUpdateMs >= 250) {
        lastDisplayUpdateMs = millis();
        showPulseScreen("MEASURE HEART RATE");
    }
}

// ------------------------- MPU6050 INITIALIZATION AND ACTIVITY -------------------------
// MPU6050 register-level driver. This deliberately avoids Adafruit Unified
// Sensor because its sensor_t typedef conflicts with esp_camera's sensor_t.
static constexpr uint8_t MPU6050_ADDR_LOW = 0x68;
static constexpr uint8_t MPU6050_ADDR_HIGH = 0x69;
static uint8_t mpuAddress = MPU6050_ADDR_LOW;

static bool mpuWriteRegister(uint8_t reg, uint8_t value) {
    Wire.beginTransmission(mpuAddress);
    Wire.write(reg);
    Wire.write(value);
    return Wire.endTransmission() == 0;
}

static bool mpuReadRegisters(uint8_t startReg, uint8_t *buffer, size_t length) {
    Wire.beginTransmission(mpuAddress);
    Wire.write(startReg);
    if (Wire.endTransmission(false) != 0) return false;

    const size_t received = Wire.requestFrom((uint8_t)mpuAddress, (uint8_t)length, (uint8_t)true);
    if (received != length) return false;

    for (size_t i = 0; i < length; ++i) buffer[i] = Wire.read();
    return true;
}

static int16_t combineBytes(uint8_t highByte, uint8_t lowByte) {
    return (int16_t)((uint16_t)highByte << 8 | lowByte);
}

static bool initMpu6050() {
    // Check the two standard MPU6050 I2C addresses.
    mpuAddress = MPU6050_ADDR_LOW;
    Wire.beginTransmission(mpuAddress);
    if (Wire.endTransmission() != 0) {
        mpuAddress = MPU6050_ADDR_HIGH;
        Wire.beginTransmission(mpuAddress);
        if (Wire.endTransmission() != 0) {
            Serial.println("[MPU6050] not detected at 0x68 or 0x69");
            return false;
        }
    }

    // Wake device; select internal 8 MHz oscillator.
    if (!mpuWriteRegister(0x6B, 0x00)) return false;
    delay(100);

    // DLPF_CFG=3 (44 Hz accel / 42 Hz gyro), +/-4g, +/-500 degrees/sec.
    if (!mpuWriteRegister(0x1A, 0x03)) return false;
    if (!mpuWriteRegister(0x1C, 0x08)) return false;
    if (!mpuWriteRegister(0x1B, 0x08)) return false;
    if (!mpuWriteRegister(0x19, 0x04)) return false; // 200 Hz sample rate

    uint8_t whoAmI = 0;
    if (!mpuReadRegisters(0x75, &whoAmI, 1)) return false;
    Serial.printf("[MPU6050] detected at 0x%02X, WHO_AM_I=0x%02X\n",
                  mpuAddress, whoAmI);

    // 0x68 = MPU6050; 0x74 = MPU6515, a closely related 6-axis
    // device with compatible accel/gyro output registers and scale factors.
    if (whoAmI == 0x68) {
        Serial.println("[MPU6050] MPU6050 identity confirmed");
        return true;
    }
    if (whoAmI == 0x74) {
        Serial.println("[MPU6050] MPU6515-compatible identity detected; using shared accel/gyro register map");
        return true;
    }

    Serial.printf("[MPU6050] unsupported WHO_AM_I value: 0x%02X\n", whoAmI);
    return false;
}

static void resetRhythmProgress(const char *reason) {
    if (rhythmActive || accumulatedMovementMs > 0) {
        Serial.printf("[MPU6050] rhythm reset: %s\\n", reason);
    }
    accumulatedMovementMs = 0;
    lastRhythmPeakMs = 0;
    rhythmStartMs = 0;
    rhythmIntervalCount = 0;
    rhythmActive = false;
    activitySignalArmed = true;
}

static void pollMpu6050() {
    const uint32_t now = millis();
    if (lastMpuSampleMs != 0 && now - lastMpuSampleMs < kMpuSampleIntervalMs) return;
    lastMpuSampleMs = now;

    uint8_t raw[14];
    if (!mpuReadRegisters(0x3B, raw, sizeof(raw))) {
        movementDetectedThisSample = false;
        if (lastValidMpuSampleMs != 0 &&
            now - lastValidMpuSampleMs > kRhythmGapTimeoutMs) {
            resetRhythmProgress("I2C sample gap");
        }
        Serial.println("[MPU6050] register read failed; sample ignored");
        return;
    }

    lastValidMpuSampleMs = now;

    const int16_t rawAx = combineBytes(raw[0], raw[1]);
    const int16_t rawAy = combineBytes(raw[2], raw[3]);
    const int16_t rawAz = combineBytes(raw[4], raw[5]);
    const int16_t rawGx = combineBytes(raw[8], raw[9]);
    const int16_t rawGy = combineBytes(raw[10], raw[11]);
    const int16_t rawGz = combineBytes(raw[12], raw[13]);

    // +/-4g = 8192 LSB/g; +/-500 deg/s = 65.5 LSB/(deg/s).
    const float ax = (rawAx / 8192.0f) * 9.80665f;
    const float ay = (rawAy / 8192.0f) * 9.80665f;
    const float az = (rawAz / 8192.0f) * 9.80665f;
    const float gx = (rawGx / 65.5f) * (PI / 180.0f);
    const float gy = (rawGy / 65.5f) * (PI / 180.0f);
    const float gz = (rawGz / 65.5f) * (PI / 180.0f);

    const float accelerationMagnitude = sqrtf(ax * ax + ay * ay + az * az);
    latestGyroMagnitude = sqrtf(gx * gx + gy * gy + gz * gz);

    if (!gravityEstimateReady) {
        gravityMagnitudeEstimate = accelerationMagnitude;
        gravityEstimateReady = true;
    } else {
        gravityMagnitudeEstimate += kGravityFilterAlpha *
                                    (accelerationMagnitude - gravityMagnitudeEstimate);
    }

    // Removing the slowly changing gravity component makes this less sensitive
    // to how the handheld sensor is oriented.
    latestDynamicAcceleration = fabsf(accelerationMagnitude - gravityMagnitudeEstimate);
    const float gyroActivity = latestGyroMagnitude * kGyroToLinearFactor;
    const float activitySignal = fmaxf(latestDynamicAcceleration, gyroActivity);
    filteredActivitySignal += kActivityFilterAlpha *
                              (activitySignal - filteredActivitySignal);

    movementDetectedThisSample = filteredActivitySignal >= kLinearPeakThreshold;

    // A peak is counted only after the signal has first returned below the
    // lower hysteresis threshold. This prevents one long shake from counting
    // repeatedly as multiple movement cycles.
    if (filteredActivitySignal <= kLinearResetThreshold) {
        activitySignalArmed = true;
    }

    if (activitySignalArmed && filteredActivitySignal >= kLinearPeakThreshold) {
        activitySignalArmed = false;

        if (lastRhythmPeakMs == 0) {
            lastRhythmPeakMs = now;
            rhythmIntervalCount = 0;
        } else {
            const uint32_t interval = now - lastRhythmPeakMs;

            if (interval >= kRhythmMinIntervalMs &&
                interval <= kRhythmMaxIntervalMs) {
                lastRhythmPeakMs = now;
                ++rhythmIntervalCount;

                if (!rhythmActive &&
                    rhythmIntervalCount >= kMinimumIntervalsToStart) {
                    rhythmActive = true;
                    rhythmStartMs = now;
                    accumulatedMovementMs = 0;
                    Serial.println("[MPU6050] sustained rhythm confirmed; 25-second timer started");
                } else if (rhythmActive) {
                    accumulatedMovementMs = now - rhythmStartMs;
                }
            } else {
                // An early/late peak breaks the cadence. Treat it as the
                // beginning of a new candidate sequence, not as progress.
                lastRhythmPeakMs = now;
                rhythmIntervalCount = 0;
                rhythmActive = false;
                accumulatedMovementMs = 0;
                rhythmStartMs = 0;
            }
        }
    }

    // A missing beat means the user stopped or the cadence was lost.
    if (lastRhythmPeakMs != 0 &&
        now - lastRhythmPeakMs > kRhythmGapTimeoutMs) {
        resetRhythmProgress("rhythmic movement stopped");
    }

    Serial.printf("[MPU6050] dyn=%.2f gyro=%.2f signal=%.2f rhythm=%s intervals=%lu progress=%lu ms\\n",
                  latestDynamicAcceleration, latestGyroMagnitude,
                  filteredActivitySignal, rhythmActive ? "YES" : "NO",
                  (unsigned long)rhythmIntervalCount,
                  (unsigned long)accumulatedMovementMs);
}

static void updateActivityDisplay() {
    if (millis() - lastActivityDisplayMs >= 250) {
        lastActivityDisplayMs = millis();
        showActivityScreen();
    }
}

// ------------------------- ALARM / TRIGGER -------------------------
static void stopBuzzer() {
    buzzerIsOn = false;
    setBuzzer(false);
}

static void updateBuzzer() {
    const uint32_t now = millis();
    if (now - lastBuzzerToggleMs >= 300) {
        lastBuzzerToggleMs = now;
        buzzerIsOn = !buzzerIsOn;
        setBuzzer(buzzerIsOn);
    }
}

static void handleWaitingForTrigger() {
    const bool triggerReleased = digitalRead(START_TRIGGER_PIN) == HIGH;
    if (triggerReleased) {
        triggerLatched = false;
    } else if (!triggerLatched) {
        triggerLatched = true;
        Serial.println("[TRIGGER] Jumper trigger received");
        enterState(ALARM_RINGING);
    }
}

// ------------------------- SETUP -------------------------
void setup() {
    Serial.begin(115200);
    delay(1200);
    Serial.println();
    Serial.println("=== SMART ALARM CLOCK: ESP32-CAM + TinyML + MAX30102 ===");
    Serial.printf("[SYS] chip=%s PSRAM=%s (%u bytes free)\n",
                  ESP.getChipModel(), psramFound() ? "yes" : "no",
                  static_cast<unsigned>(ESP.getFreePsram()));

    pinMode(kFlashLedPin, OUTPUT);
    digitalWrite(kFlashLedPin, LOW);

    pinMode(BUZZER_PIN, OUTPUT);
    setBuzzer(false);

    pinMode(START_TRIGGER_PIN, INPUT_PULLUP);
    // If the jumper is already fitted during reset, require it to be removed
    // before accepting a new trigger. This prevents an immediate repeat alarm.
    triggerLatched = (digitalRead(START_TRIGGER_PIN) == LOW);

    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
    Wire.setClock(100000);

    oledReady = display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDRESS);
    if (oledReady) {
        Serial.println("[OLED] initialized at 0x3C");
        showScreen("SMART ALARM CLOCK", "Starting sensors...");
    } else {
        Serial.println("[OLED] initialization failed; continuing with Serial output");
    }

    Serial.println("[MAX30102] initializing...");
    maxReady = particleSensor.begin(Wire, I2C_SPEED_STANDARD);
    if (!maxReady) {
        Serial.println("[FATAL] MAX30102 initialization failed");
        showScreen("MAX30102 ERROR", "Check sensor wiring", "I2C address should", "be 0x57");
        while (true) delay(1000);
    }

    // Use the same known-working default setup as the standalone sensor test.
    particleSensor.setup();
    particleSensor.setPulseAmplitudeRed(0x0A);
    particleSensor.setPulseAmplitudeIR(0x1F);
    particleSensor.setPulseAmplitudeGreen(0);
    Serial.println("[MAX30102] initialized successfully");

    Serial.println("[MPU6050] initializing...");
    mpuReady = initMpu6050();
    if (!mpuReady) {
        Serial.println("[FATAL] MPU6050 initialization failed");
        showScreen("MPU6050 ERROR", "Check VCC/GND/SDA/SCL", "Try I2C address", "0x68 or 0x69");
        while (true) delay(1000);
    }

    if (!initCamera()) {
        Serial.println("[FATAL] Camera initialization failed");
        showScreen("CAMERA ERROR", "Camera init failed");
        while (true) delay(1000);
    }

    if (!initModel()) {
        Serial.println("[FATAL] Person model initialization failed");
        showScreen("MODEL ERROR", "TinyML init failed");
        while (true) delay(1000);
    }

    Serial.println("[READY] Waiting for GPIO13 jumper trigger.");
    enterState(WAITING_FOR_TRIGGER);
}

// ------------------------- MAIN LOOP -------------------------
void loop() {
    switch (systemState) {
        case WAITING_FOR_TRIGGER:
            handleWaitingForTrigger();
            break;

        case ALARM_RINGING: {
            updateBuzzer();
            bool confirmedPerson = false;
            runPersonDetection(confirmedPerson);

            if (confirmedPerson) {
                Serial.println("[PERSON] Confirmed by 3 consecutive frames. Stopping alarm.");
                stopBuzzer();
                digitalWrite(kFlashLedPin, LOW);
                beginHeartRateMeasurement();
            }
            break;
        }

        case MEASURE_HEART_RATE:
            if (pollPulseSensor()) {
                acceptedHeartRateBpm = bpmSum / static_cast<float>(validBpmSamples);
                Serial.printf("[PULSE] Accepted heart rate: %.1f BPM\n", acceptedHeartRateBpm);
                enterState(DETECT_ACTIVITY);
            } else {
                updatePulseUi();
            }
            break;

        case DETECT_ACTIVITY:
            pollMpu6050();
            updateActivityDisplay();

            if (accumulatedMovementMs >= kRequiredMovementMs) {
                Serial.printf("[MPU6050] activity requirement met: %lu ms accumulated movement\n",
                              (unsigned long)accumulatedMovementMs);
                enterState(ROUTINE_COMPLETE);
            } else if (millis() - activityStartMs >= kActivityReminderMs) {
                Serial.println("[MPU6050] still waiting for sufficient movement; continuing detection");
                activityStartMs = millis();
            }
            break;

        case ROUTINE_COMPLETE:
            if (millis() - stateStartMs >= kDoneScreenMs) {
                enterState(WAITING_FOR_TRIGGER);
            }
            break;
    }
}
