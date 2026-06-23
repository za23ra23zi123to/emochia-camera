import cv2
from deepface import DeepFace
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Emotion Detector. Press Q to exit")
frame_count = 0
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if frame_count % 5 == 0:
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            confidence = result[0]['emotion'][emotion]
            if confidence > 50:
                cv2.putText(frame, f"{emotion.upper()} ({confidence:.0f}%)", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "LOW CONFIDENCE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        except:
            cv2.putText(frame, "NO FACE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    frame_count += 1
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()