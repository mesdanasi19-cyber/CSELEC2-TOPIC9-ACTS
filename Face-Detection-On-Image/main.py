import os
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(SCRIPT_DIR, 'haarcascade_frontalface_default.xml')
image_path = os.path.join(SCRIPT_DIR, 'woodcutters.jpg')

# Try using the system path data first
fallback_xml = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

# 1. Load the pre-trained Haar Cascade XML model
if os.path.exists(xml_path):
    face_cascade = cv2.CascadeClassifier(xml_path)
elif os.path.exists(fallback_xml):
    face_cascade = cv2.CascadeClassifier(fallback_xml)
else:
    print("Error: Could not find the XML template anywhere on your machine.")
    print(f"Please copy 'haarcascade_frontalface_default.xml' manually into:\n{SCRIPT_DIR}")
    exit()

# Verify the model loaded into memory
if face_cascade.empty():
    print("Error: The cascade file found is corrupted or empty.")
    exit()

# 2. Read the image
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Could not read image at:\n{image_path}")
    print("Make sure 'woodcutters.jpg' is placed inside the 'Face-Detection-On-Image' folder!")
    exit()

# 3. Detect and display
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.08, 5)

for x, y, w, h in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

cv2.imshow('Detected Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
