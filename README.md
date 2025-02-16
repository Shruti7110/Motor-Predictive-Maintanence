# Motor-Predictive-Maintanence

# Predictive Maintenance with Machine Learning
A machine learning model to detect motor anomalies using real-time accelerometer and gyroscope data.

# Training data using Neural network
Add your data collected when motor is in ideal state and during abnormal conditions to the ML_Model_train.py file. (Feel free to use sample data provided to try out the code).
The script will train based on the data and generate a model.cpp file which you will need to add to the arduino repository

# Deploying your model on using tensorflow lite on arduino
In a same folder as your Arduino_Motor_state_prediction.ino file add constants.h, arduino_constants.cpp, model.h and model.cpp (The file which was generated on running the above python script).
Now run the code, you can see the state of your motor. Feel free to customize the code to glow LED when anamoly detected and so now.

## Contact
Created by (https://github.com/Shruti7110) - feel free to reach out!

## Acknowledgments
Thanks to OpenAI, Scikit-Learn, and TensorFlow for inspiration.
