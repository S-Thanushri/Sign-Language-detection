import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained sign language recognition model
model = tf.keras.models.load_model(file_path)  # Replace with your model file path

# Define a list of class labels corresponding to the signs the model can recognize
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define the coordinates for your ROI (region of interest)
roi_x = 100
roi_y = 100
roi_width = 200
roi_height = 200

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Extract the ROI from the frame
    roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]

    # Convert the ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Preprocess the grayscale image (resize, normalize, etc.)
    resized_roi = cv2.resize(gray_roi, (28, 28))  # Resize to 28x28 pixels
    normalized_roi = resized_roi / 255.0  # Normalize pixel values to the range [0, 1]
    input_data = np.expand_dims(normalized_roi, axis=0)  # Add a batch dimension
    input_data = np.expand_dims(input_data, axis=-1)  # Add a channel dimension

    # Perform sign language recognition
    predictions = model.predict(input_data)
    predicted_class = class_labels[np.argmax(predictions)]

    # Display the prediction on the frame
    cv2.putText(frame, f"Sign: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw a rectangle around the ROI
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_width, roi_y+roi_height), (0, 255, 0), 2)

    # Display the frame with the prediction
    cv2.imshow('Sign Language Recognition', frame)

    # Exit the program when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
