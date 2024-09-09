import cv2
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, 90), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Gender: {dominant_gender}", (10, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Age: {str(age)}", (10, 130), font, 1, (0, 0, 255), 2, cv2.LINE_AA)


    if dominant_gender.lower() == 'woman' and dominant_emotion.lower() == 'fear':
        cv2.putText(frame, "Alert: Woman in Fear!", (10, 170), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('demo video', frame)


    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
