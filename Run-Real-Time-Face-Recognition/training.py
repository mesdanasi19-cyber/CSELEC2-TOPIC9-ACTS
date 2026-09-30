import os
import cv2
import numpy as np


def read_images(path, image_size=(200, 200)):
  names = []
  images, labels = [], []
  label = 0

  # Get sorted list of subdirectories to maintain consistent label ordering
  subdirs = sorted([
      d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
  ])

  for subdirname in subdirs:
    names.append(subdirname)
    subject_path = os.path.join(path, subdirname)

    for filename in os.listdir(subject_path):
      img_path = os.path.join(subject_path, filename)
      img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
      if img is None:
        continue
      img = cv2.resize(img, image_size)
      images.append(img)
      labels.append(label)

    label += 1

  return names, np.asarray(images), np.asarray(labels)


# Load training data from 'faces' directory
path = 'faces'
names, images, labels = read_images(path)

# Create and train the LBPH Face Recognizer
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

# Save the trained model weights
model.save('face_model.yml')
print(f'✔ Model successfully trained on {len(names)} classes: {names}')
print('✔ Saved as face_model.yml')