// Include necessary libraries
#include <Arduino_BMI270_BMM150.h> // IMU sensor library
#include <TensorFlowLite.h> // TensorFlow Lite for Microcontrollers
#include "model.h" // Neural network model data
#include "constants.h" // Constants for the model
#include "tensorflow/lite/micro/micro_interpreter.h" // Interpreter
#include "tensorflow/lite/micro/micro_log.h" // Debugging utility
#include "tensorflow/lite/schema/schema_generated.h" // Schema definitions
#include "tensorflow/lite/micro/all_ops_resolver.h" // TFLite operations resolver
#include "tensorflow/lite/micro/system_setup.h" // System setup for TFLite
#include <math.h> // Math functions

// TensorFlow Lite model and interpreter
namespace {
    tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;

    constexpr int kTensorArenaSize = 60 * 1024; // Allocate memory for model inference
    alignas(16) uint8_t tensor_arena[kTensorArenaSize]; // Memory buffer
}

// Define buffer size for storing IMU data
const int N = 100;
float ax_buffer[N], ay_buffer[N], az_buffer[N];
float gx_buffer[N], gy_buffer[N], gz_buffer[N];
int num = 0;
bool buffer_full = false;
#define EPSILON 1e-6

void setup() {
    tflite::InitializeTarget();

    // Load pre-trained TensorFlow model
    model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model schema version mismatch: %d != %d", model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // Set up operation resolver
    static tflite::AllOpsResolver micro_op_resolver;

    // Create an interpreter instance
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Allocate memory for model tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed");
        return;
    }

    // Get input tensor
    input = interpreter->input(0);
    if (input->dims->size != 2 || input->dims->data[0] != 1 || input->dims->data[1] != 24 || input->type != kTfLiteFloat32) {
        MicroPrintf("Bad input tensor parameters in model");
        return;
    }

    // Initialize IMU sensor
    if (!IMU.begin()) {
        Serial.println("Failed to initialize IMU!");
        delay(2000);
        while (1);
    }
    Serial.println("Arduino ML Model Ready for Deployment!");
}

// Function to compute softmax activation
void softmax(float* input, float* output) {
    float max_value = input[0];
    for (int i = 0; i < 2; i++) if (input[i] > max_value) max_value = input[i];
    float sum = 0.0;
    for (int i = 0; i < 2; i++) {
        output[i] = exp(input[i] - max_value);
        sum += output[i];
    }
    for (int i = 0; i < 2; i++) output[i] /= sum;
}

// Statistical feature extraction functions
float compute_mean(float* buffer) {
    float sum = 0.0;
    for (int i = 0; i < N; i++) sum += buffer[i];
    return sum / N;
}

float compute_std(float* buffer, float mean) {
    float variance = 0.0;
    for (int i = 0; i < N; i++) variance += pow(buffer[i] - mean, 2);
    return sqrt(variance / (N - 1));
}

float compute_max(float* buffer) {
    float max_value = buffer[0];
    for (int i = 0; i < N; i++) if (buffer[i] > max_value) max_value = buffer[i];
    return max_value;
}

float compute_min(float* buffer) {
    float min_value = buffer[0];
    for (int i = 0; i < N; i++) if (buffer[i] < min_value) min_value = buffer[i];
    return min_value;
}

void loop() {
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
        float ax, ay, az, gx, gy, gz;
        IMU.readAcceleration(ax, ay, az);
        IMU.readGyroscope(gx, gy, gz);

        // Store sensor data in buffers
        ax_buffer[num] = ax;
        ay_buffer[num] = ay;
        az_buffer[num] = az;
        gx_buffer[num] = gx;
        gy_buffer[num] = gy;
        gz_buffer[num] = gz;
        num++;

        if (num >= N) {
            buffer_full = true;
            num = 0;
        }
    

        if (buffer_full) {
            buffer_full = false;

            // Compute statistical features for each axis
            float features[24];
            int idx = 0;
            for (int i = 0; i < 6; i++) {
              float* buffer = (i == 0) ? ax_buffer : (i == 1) ? ay_buffer : (i == 2) ? az_buffer : (i == 3) ? gx_buffer : (i == 4) ? gy_buffer : gz_buffer;
              float mean = compute_mean(buffer);
              features[idx++] = mean;
              features[idx++] = compute_std(buffer, mean);
              features[idx++] = compute_max(buffer);
              features[idx++] = compute_min(buffer);
            }

            // Standardizing and feeding data into the model
            for (int i = 0; i < 24; i++) {
              input->data.f[i] = (features[i] - mean_scaler[i]) / std_scaler[i];
            }             
            // Get inference results
            float logits[2] = { output->data.f[0], output->data.f[1] };
            float probability[2];
            softmax(logits, probability);
            Serial.print("Motor state: ");
            if (probability[0] > 0.75) Serial.println("Normal");
            else Serial.println("Anomaly Detected");

            // Run model inference
            if (interpreter->Invoke() != kTfLiteOk) {
                Serial.println("Model invocation failed");
                return;
            }
        }
    }      
        
    delay(10);
}
