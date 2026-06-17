import cv2
from deepface import DeepFace
import os

# Инициализация камеры
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Загружаем каскад из папки с проектом
cascade_path = 'haarcascade_frontalface_default.xml'

if not os.path.exists(cascade_path):
    print(f"Файл {cascade_path} не найден в папке проекта!")
    exit()

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить каскад!")
    exit()

print("Detector emocionov zapushchen. Nazhmite Q dlya vyhoda.")
print("Dlya luchshego raspoznavaniya smotrite pryamo v kameru.")

# Для сглаживания результатов
emotion_history = []
history_size = 5

# Текущая отображаемая эмоция
current_emotion = "WAIT"
current_confidence = 0

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # УЛУЧШЕННАЯ ДЕТЕКЦИЯ: более чувствительные параметры
    # minNeighbors=3 - меньше требований к соседям (ловит больше лиц)
    # scaleFactor=1.05 - более точный поиск
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(80, 80))

    # Если лиц не найдено, пробуем с другими параметрами
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(60, 60))

    if len(faces) > 0:
        # Сортируем лица по размеру (от большего к меньшему)
        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

        # Показываем все найденные лица
        for i, (x, y, w, h) in enumerate(faces_sorted):
            # Рисуем рамку для каждого лица
            if i == 0:
                # Первое лицо (самое большое) - зеленым
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                # Остальные лица - синим
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"Face {i + 1}", (x, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Берем самое большое лицо для анализа эмоций
        (x, y, w, h) = faces_sorted[0]

        # Анализируем эмоции каждый 2-й кадр (чаще для лучшей реакции)
        if frame_count % 2 == 0:
            try:
                # Увеличиваем область лица для лучшего анализа
                pad = int(min(w, h) * 0.15)
                y1 = max(0, y - pad)
                y2 = min(frame.shape[0], y + h + pad)
                x1 = max(0, x - pad)
                x2 = min(frame.shape[1], x + w + pad)
                face_roi_expanded = frame[y1:y2, x1:x2]

                result = DeepFace.analyze(
                    face_roi_expanded,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv'
                )

                emotion = result[0]['dominant_emotion']
                confidence = result[0]['emotion'][emotion]

                emotion_history.append((emotion, confidence))
                if len(emotion_history) > history_size:
                    emotion_history.pop(0)

                if len(emotion_history) > 0:
                    emotions_list = [e[0] for e in emotion_history]
                    most_common = max(set(emotions_list), key=emotions_list.count)
                    avg_confidence = 0
                    for e, c in emotion_history:
                        if e == most_common:
                            avg_confidence = c
                            break

                    # Понижаем порог уверенности для более частого отображения
                    if avg_confidence > 40:
                        current_emotion = most_common.upper()
                        current_confidence = avg_confidence
                    else:
                        current_emotion = "NE UVEREN"
                        current_confidence = avg_confidence

            except Exception as e:
                current_emotion = "OSHIBKA"
                current_confidence = 0

        # Отображаем эмоцию для самого большого лица
        if current_emotion and current_emotion != "WAIT":
            # Добавляем эмодзи для наглядности
            emoji_map = {
                'HAPPY': '😊',
                'SAD': '😢',
                'ANGRY': '😠',
                'SURPRISE': '😲',
                'FEAR': '😨',
                'DISGUST': '🤢',
                'NEUTRAL': '😐'
            }
            emoji = emoji_map.get(current_emotion, '')
            text = f"{emoji} {current_emotion} ({current_confidence:.0f}%)" if current_confidence > 0 else current_emotion
            cv2.putText(frame, text, (x, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Показываем количество лиц
        cv2.putText(frame, f"Lits: {len(faces_sorted)}", (x, y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    else:
        # Лицо не найдено - пробуем улучшить освещение
        cv2.putText(frame, "LITSO NE NAJDENO", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, "Poprobuyte svetlee ili smotrite pryamo", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if len(emotion_history) > 0:
            emotion_history.clear()
            current_emotion = "WAIT"

    # Информация внизу
    cv2.putText(frame, f"Kadrov: {frame_count} | Nazhmite Q dlya vyhoda", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    frame_count += 1
    cv2.imshow('Detector emociy', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Programma zavershena.")