from PIL import Image
import os
# Define resize function
def resize_image(image_path, output_path, size=(128, 128)):
    img = Image.open(image_path)
    img = img.resize(size)
    img.save(output_path)


# Resize images from cropped_dataset to resized_dataset
input_folder = 'cropped_dataset/facedatas/'
output_folder = 'resized_dataset/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for subfolder in os.listdir(input_folder):
    subfolder_path = os.path.join(input_folder, subfolder)
    output_subfolder = os.path.join(output_folder, subfolder)

    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    # Loop through files in each subfolder
    for filename in os.listdir(subfolder_path):
        input_path = os.path.join(subfolder_path, filename)
        output_path = os.path.join(output_subfolder, filename)

        # Check if the path is a file before trying to open it
        if os.path.isfile(input_path):
            resize_image(input_path, output_path)
