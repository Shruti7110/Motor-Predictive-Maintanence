
# Import necessary libraries
import pandas as pd  # For handling and manipulating data
import numpy as np  # For numerical operations

# Machine learning models and utilities
from sklearn.model_selection import train_test_split, GridSearchCV  # For splitting data and hyperparameter tuning
from sklearn.ensemble import RandomForestClassifier  # Random Forest classifier
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.neural_network import MLPClassifier  # Multi-layer Perceptron Neural Network
from sklearn.preprocessing import StandardScaler  # For data standardization
from sklearn.metrics import accuracy_score  # To evaluate model performance

# Serial communication for real-time data acquisition
#import serial.tools.list_ports  # To list available COM ports
from collections import deque  # To manage a buffer for real-time data
import joblib  # To save and load trained models
import tensorflow as tf  # For deep learning
from tensorflow.keras.models import Sequential  # Keras model creation
from tensorflow.keras.layers import Dense  # Neural network layers
import binascii

# Load Excel data into Pandas DataFrame
ideal_df = pd.read_csv("sensor_dump_data_normal.csv")  # Data when motor is in normal condition
anomaly_df = pd.read_csv("sensor_dump_data_anomaly.csv")  # Data when anomaly is detected

# Label the data
ideal_df['Label'] = 0  # 0 represents normal motor state
anomaly_df['Label'] = 1  # 1 represents anomaly detected

# Merge both datasets into one
data = pd.concat([ideal_df, anomaly_df], ignore_index=True)  # Combine datasets row-wise

# Feature Engineering: Extract statistical features from accelerometer & gyroscope data
def extract_features(df):
    features = pd.DataFrame()
    for axis in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']:  # Accelerometer & Gyroscope axes
        features[f'{axis}_mean'] = df[axis].mean()
        features[f'{axis}_std'] = df[axis].std()
        features[f'{axis}_max'] = df[axis].max()
    return features

# Apply feature extraction in a sliding window of 100 samples
feature_data = pd.DataFrame()
labels = []
for i in range(0, len(data), 100):  # Sliding window step of 100
    window = data.iloc[i:i+100]
    feature_data = feature_data.append(extract_features(window), ignore_index=True)
    labels.append(window['Label'].iloc[-1])  # Assign last label of the window

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(feature_data, labels, test_size=0.2, random_state=42)

# Standardize data for better ML performance
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Define models and hyperparameter tuning
param_grid = {
    'hidden_layer_sizes': [(50,50), (100,50), (100,100)],
    'max_iter': [500, 1000],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd']
}
mlp_grid = GridSearchCV(MLPClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
mlp_grid.fit(x_train, y_train)
mlp_best = mlp_grid.best_estimator_

# Define machine learning model (Neural Network)
y_pred = mlp_best.predict(x_test)
print(f"MLP Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save trained model
joblib.dump(mlp_best, "motor_nn_model.pkl")

# Real-time data acquisition from Arduino
def get_real_time_data():
    ''' function to check if the model is working correctly by printing the prediction
    on the console before sending the data to the arduino as a package'''
    #ports =serial.tools.list_ports.comports()
    ser = serial.Serial('COM12', 9600, timeout=1)  # Open COM port
    ser.open()
    buffer = deque(maxlen=100)  # Sliding window buffer
    try:
        while True:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    values = list(map(float, line.split(',')))
                    if len(values) == 6:
                        buffer.append(values)
                        if len(buffer) == 100:
                            real_time_features = extract_features(pd.DataFrame(buffer, columns=['ax', 'ay', 'az', 'gx', 'gy', 'gz']))
                            real_time_features = scaler.transform(real_time_features)
                            prediction = mlp_best.predict(real_time_features)[0]
                            state_map = {0: "Normal", 1: "Anamoly"}
                            print(f"Motor State: {state_map[prediction]}")
    except KeyboardInterrupt:
        print("Stopping real time monitoring")
        ser.close()

# Convert the trained model to TensorFlow format
def convert_sklearn_mlp_to_keras(mlp_model):
    keras_model = Sequential()
    keras_model.add(Dense(50, activation='relu', input_shape=(x_train.shape[1],)))
    for layer_size in mlp_model.hidden_layer_sizes[1:]:
        keras_model.add(Dense(50, activation='relu'))
    keras_model.add(Dense(2, activation='softmax'))  # Output layer for classification
    keras_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return keras_model

keras_model = convert_sklearn_mlp_to_keras(mlp)
keras_model.save("motor_nn_model.keras")  # Save model in Keras format

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('motor_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Convert to C array for Arduino
def convert_tflite_to_c_array():
    with open('motor_model.tflite', 'rb') as f:
        hex_data = binascii.hexlify(f.read()).decode('ascii')
    c_array = '#include "model.h" \n'
    c_array +='alignas(16) const unsigned char model_data[] = {'
    for i in range(0, len(hex_data), 2):
        c_array += '0x' + hex_data[i:i+2] + ', '
    c_array = c_array[:-2] + '};\n'
    c_array += 'const int model_len = {};' .format(len(hex_data)//2)
    with open('model.cpp', 'w') as f:
        f.write(c_array)
    print('C header file created successfully')

convert_tflite_to_c_array()

print('Model conversion completed!')

