import os
from mtcnn import MTCNN
import cv2

detector = MTCNN()

# Detect and crop faces
def detect_and_crop_faces(image_path, output_path):
    img = cv2.imread(image_path)
    faces = detector.detect_faces(img)
    if faces:
        x, y, width, height = faces[0]['box']
        cropped_face = img[y:y+height, x:x+width]
        cv2.imwrite(output_path, cropped_face)

input_folder = 'faceiD/facedatas/'
output_folder = 'cropped_dataset/facedatas/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all subfolders in facedatas
for subfolder in os.listdir(input_folder):
    subfolder_path = os.path.join(input_folder, subfolder)

    # Make sure it is a directory
    if os.path.isdir(subfolder_path):
        output_subfolder = os.path.join(output_folder, subfolder)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        for filename in os.listdir(subfolder_path):
            input_path = os.path.join(subfolder_path, filename)
            output_path = os.path.join(output_subfolder, filename)
            detect_and_crop_faces(input_path, output_path)
