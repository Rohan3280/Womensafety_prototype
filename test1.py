import cv2
from deepface import DeepFace

# Load the face detection model
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    try:
        # Analyze the frame using DeepFace
        result = DeepFace.analyze(frame, actions=['emotion', 'age', 'gender'], enforce_detection=False)
        if len(result) > 0:
            result = result[0]  # Get the first result if multiple faces are detected
            dominant_emotion = result.get('dominant_emotion', 'Unknown')
            dominant_gender = result.get('dominant_gender', 'Unknown')
            age = result.get('age', 'Unknown')
        else:
            dominant_emotion = 'Unknown'
            dominant_gender = 'Unknown'
            age = 'Unknown'
    except Exception as e:
        print(f"DeepFace error: {e}")
        dominant_emotion = 'Unknown'
        dominant_gender = 'Unknown'
        age = 'Unknown'

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the dominant emotion on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, dominant_emotion, (10, 90), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, dominant_gender, (10, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, str(age), (10, 130), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the frame with annotations
    cv2.imshow('demo video', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
