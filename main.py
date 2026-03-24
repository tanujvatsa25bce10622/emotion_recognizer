import cv2
from deepface import DeepFace

# Initialize webcam
camera = cv2.VideoCapture(0)

print("🎥 Starting real-time emotion recognition. Press 'q' to quit.")

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Detect emotions using DeepFace
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']

        # Overlay emotion text on frame
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except Exception as e:
        print("Detection error:", e)
        dominant_emotion = "Unknown"

    # Show video window
    cv2.imshow('Real-Time Emotion Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
