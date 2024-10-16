import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('drowsiness_model.h5')

# Function to preprocess the image
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    return np.expand_dims(frame, axis=0)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = preprocess_frame(frame)
    
    # Make predictions
    predictions = model.predict(processed_frame)
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index]

    # Define class names
    classes = ['Awake', 'Drowsy']

    # Display results
    label = f"{classes[class_index]}: {confidence:.2f}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Drowsiness Detection', frame)

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()