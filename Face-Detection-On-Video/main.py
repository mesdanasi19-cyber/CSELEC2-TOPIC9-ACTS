import cv2

# Load models for faces and eyes
face_cascade = cv2.CascadeClassifier(
    f'{cv2.data.haarcascades}haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    f'{cv2.data.haarcascades}haarcascade_eye.xml'
)

# Open webcam stream
camera = cv2.VideoCapture(0)

while True:
  success, frame = camera.read()
  if not success:
    break

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)

  for x, y, w, h in faces:
    # Draw blue rectangle around face
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Crop Region of Interest (ROI) for the face in grayscale
    roi_gray = gray[y : y + h, x : x + w]

    # Detect eyes ONLY within the face region
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
    for ex, ey, ew, eh in eyes:
      # Draw green rectangle around eyes (offset coordinates by x and y)
      cv2.rectangle(
          frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2
      )

  cv2.imshow('Face Detection', frame)
  if cv2.waitKey(1) != -1:  # Exit when any key is pressed
    break

camera.release()
cv2.destroyAllWindows()