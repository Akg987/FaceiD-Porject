import cv2
import torch
from PIL import Image
from mtcnn import MTCNN
import torch.nn as nn
import torchvision.models as models
import numpy as np

# Load the trained PyTorch model
num_classes = 1804
model = models.resnet18(weights=None)  # Load without pre-trained weights
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('face_recognition_model_pytorch.pth'))  # Load the trained weights
model.eval()

# MTCNN for face detection
detector = MTCNN()

# OpenCV to capture video from webcam
cap = cv2.VideoCapture(0)

# Label to class name mapping
class_names = {i: f"Person_{i}" for i in range(num_classes)}

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)

    for face in faces:
        x, y, width, height = face['box']
        face_img = rgb_frame[y:y + height, x:x + width]
        face_img_pil = Image.fromarray(face_img)

        # Convert image to tensor and normalize (input should be float32)
        face_tensor = torch.tensor(np.array(face_img_pil) / 255, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            outputs = model(face_tensor)
            _, predicted = torch.max(outputs, 1)

        label = class_names[predicted.item()]
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
