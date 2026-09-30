import cv2
# 1. Load the pre-trained Haar Cascade XML model for frontal face detection
face_cascade = cv2.CascadeClassifier(
    f'{cv2.data.haarcascades}haarcascade_frontalface_default.xml'
)

# 2. Read the image and convert it to Grayscale
img = cv2.imread('woodcutters.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Detect faces of various sizes in the grayscale image
faces = face_cascade.detectMultiScale(gray, 1.08, 5)

# 4. Draw bounding boxes around detected faces
for x, y, w, h in faces:
  cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

# 5. Display the result
cv2.imshow('Detected Faces', img)
cv2.waitKey(0)