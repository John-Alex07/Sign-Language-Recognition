import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

# Load the trained model
model = load_model('ASL_model.h5')
detector = HandDetector(maxHands=1, detectionCon=0.5)

# Define the labels (adjust this based on your model's output)
categories = {  0: "0",
                1: "1",
                2: "2",
                3: "3",
                4: "4",
                5: "5",
                6: "6",
                7: "7",
                8: "8",
                9: "9",
                10: "a",
                11: "b",
                12: "c",
                13: "d",
                14: "e",
                15: "f",
                16: "g",
                17: "h",
                18: "i",
                19: "j",
                20: "k",
                21: "l",
                22: "m",
                23: "n",
                24: "o",
                25: "p",
                26: "q",
                27: "r",
                28: "s",
                29: "t",
                30: "u",
                31: "v",
                32: "w",
                33: "x",
                34: "y",
                35: "z",
            }

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define the image size and batch size
image_size = 200

def detect_hand(frame):
    # Detect hands
    hands, img = detector.findHands(frame)
    
    if hands:
        hand = hands[0]
        bbox = hand['bbox']  # Bounding box info x, y, w, h
        x, y, w, h = bbox
        
        # Add padding to the bounding box
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        
        # Crop the frame to the bounding box with padding
        hand_img = frame[y:y+h, x:x+w]
        
        return hand_img, (x, y, w, h)
    return None, None

def color_background_black(frame, bbox):
    x, y, w, h = bbox
    hand_img = frame[y:y+h, x:x+w]

    # Create a mask for the hand
    mask = np.zeros_like(hand_img)
    mask[:, :] = 255  # Set the mask to white

    # Apply the mask to the hand image
    hand_img = cv2.bitwise_and(hand_img, mask)

    # Set the background to black
    frame[y:y+h, x:x+w] = hand_img

    return frame

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Detect hand
    hand, bbox = detect_hand(frame)
    if hand is not None:
        # Color the background of the hand as black
        frame = color_background_black(frame, bbox)
        
        # Save the hand image for debugging
        cv2.imwrite('hand.png', hand)
        
        # Preprocess the hand image
        img = cv2.resize(hand, (image_size, image_size))  # Resize to the input size expected by the model
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if your model expects grayscale images
        img = img / 255.0  # Normalize the image
        img = np.expand_dims(img, axis=-1)  # Add channel dimension if needed
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Make predictions
        predictions = model.predict(img)
        predicted_label = categories[np.argmax(predictions)]
        print(f'Predicted label: {predicted_label}')
        # print(f'Predictions: {predictions}')  # Print the raw predictions for debugging

        # Display the predictions on the frame
        cv2.putText(frame, f'Prediction: {predicted_label}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Sign Language Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()