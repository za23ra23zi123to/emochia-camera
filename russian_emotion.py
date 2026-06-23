import cv2
from deepface import DeepFace
cap = cv2.VideoCapture(0)
print("Калибровка. Смотри в камеру нейтрально 3 секунды...")
neutral_emotions = []
for _ in range(30):
    ret, frame = cap.read()
    if ret:
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            neutral_emotions.append(result[0]['dominant_emotion'])
        except:
            pass
if neutral_emotions:
    your_neutral = max(set(neutral_emotions), key=neutral_emotions.count)
else:
    your_neutral = "neutral"
print(f"Твоё нейтральное лицо определено как: {your_neutral}")
print("Теперь улыбнись широко. Нажми Q для выхода")
emotion_map = {
    'happy': '😊 СЧАСТЛИВ',
    'sad': '😢 ГРУСТЬ',
    'angry': '😠 ЗЛОСТЬ',
    'surprise': '😲 УДИВЛЕНИЕ',
    'fear': '😨 СТРАХ',
    'disgust': '🤢 ОТВРАЩЕНИЕ',
    'neutral': '😐 НЕЙТРАЛЬНО'
}
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        raw_emotion = result[0]['dominant_emotion']
        confidence = result[0]['emotion'][raw_emotion]
        if raw_emotion == your_neutral and confidence > 60:
            display_emotion = "😐 НЕЙТРАЛЬНО"
        elif raw_emotion == 'happy' and confidence > 50:
            display_emotion = "😊 СЧАСТЛИВ"
        elif raw_emotion == 'sad' and confidence > 55:
            display_emotion = "😢 ГРУСТЬ"
        elif raw_emotion == 'angry' and confidence > 55:
            display_emotion = "😠 ЗЛОСТЬ"
        elif raw_emotion == 'surprise' and confidence > 55:
            display_emotion = "😲 УДИВЛЕНИЕ"
        else:
            display_emotion = "😐 НЕЙТРАЛЬНО"
        cv2.putText(frame, f"{display_emotion} ({confidence:.0f}%)", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except:
        cv2.putText(frame, "😐 ЛИЦО НЕ НАЙДЕНО", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Русский детектор эмоций', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
