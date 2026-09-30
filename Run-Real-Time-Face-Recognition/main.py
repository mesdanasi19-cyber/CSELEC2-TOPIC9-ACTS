import os
import cv2

# Load XML directly from local directory
cascade_path = os.path.join(
    os.path.dirname(__file__), 'haarcascade_frontalface_default.xml'
)
face_cascade = cv2.CascadeClassifier(cascade_path)

# Safety check to prevent !empty() crash
if face_cascade.empty():
  raise FileNotFoundError(
      f'Failed to load cascade classifier at: {cascade_path}'
  )
model = cv2.face.LBPHFaceRecognizer_create()
model.read('face_model.yml')

# Load list of character names (sorted to match trainer ordering)
names = sorted([
    d for d in os.listdir('faces') if os.path.isdir(os.path.join('faces', d))
])

camera = cv2.VideoCapture(0)
print('Press any key to exit...')

while True:
  success, frame = camera.read()
  if not success:
    break

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)

  for x, y, w, h in faces:
    face_roi = gray[y : y + h, x : x + w]
    face_resized = cv2.resize(face_roi, (200, 200))

    # Predict class label and confidence distance
    label, confidence = model.predict(face_resized)

    # LBPH Confidence: lower score = better match (< 80 is a good match threshold)
    if confidence < 80 and label < len(names):
      name = names[label]
    else:
      name = 'Unknown'

    text = f'{name} ({confidence:.1f})'
    color = (0, 255, 0) if name != 'Unknown' else (0, 0, 255)

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
    )

  cv2.imshow('Fukiko Face Recognition', frame)
  if cv2.waitKey(1) != -1:
    break

camera.release()
cv2.destroyAllWindows()