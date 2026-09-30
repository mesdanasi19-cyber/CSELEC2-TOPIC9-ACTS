import os
import cv2

# Set label name and create output folder
person_name = 'Meshach'
output_folder = f'faces/{person_name}'
os.makedirs(output_folder, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    f'{cv2.data.haarcascades}haarcascade_frontalface_default.xml'
)
camera = cv2.VideoCapture(0)
count = 0

print('Press any key to stop capturing...')

while True:
  success, frame = camera.read()
  if not success:
    break

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)

  for x, y, w, h in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Crop face, resize to standard 200x200 dimensions, and save
    face_img = cv2.resize(gray[y : y + h, x : x + w], (200, 200))
    cv2.imwrite(f'{output_folder}/{count}.pgm', face_img)
    count += 1

  cv2.imshow('Capturing Faces...', frame)
  if cv2.waitKey(1) != -1:
    break

camera.release()
cv2.destroyAllWindows()
print(f'✔ {count} images saved to {output_folder}')