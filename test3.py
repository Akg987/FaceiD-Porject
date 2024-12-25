import os
import torch
from torchvision import datasets, transforms
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
import torch.nn as nn
import torchvision.models as models

# Define transformations for PyTorch (resize, convert to tensor, normalize)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Load dataset using resized images for PyTorch
dataset = datasets.ImageFolder(root='resized_dataset/', transform=transform)

# Create DataLoader for PyTorch
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Image Data Generator for TensorFlow
datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    'resized_dataset/',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # For training set
)

validation_generator = datagen.flow_from_directory(
    'resized_dataset/',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # For validation set
)

# TensorFlow ResNet50 model setup
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Add custom layers on top
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(10, activation='softmax')(x)  # Adjust for your number of classes

# Create and compile the TensorFlow model
model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the TensorFlow model
model.fit(train_generator, validation_data=validation_generator, epochs=10)

# PyTorch ResNet18 model setup
pytorch_model = models.resnet18(pretrained=True)

# Modify the last layer for the number of classes in PyTorch
num_classes = 10  # Adjust based on your dataset
pytorch_model.fc = nn.Linear(pytorch_model.fc.in_features, num_classes)

# Define loss and optimizer for PyTorch
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.001)

# Training loop for PyTorch model
for epoch in range(10):
    for images, labels in dataloader:
        # Forward pass
        outputs = pytorch_model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/10], Loss: {loss.item()}')

# Evaluation for TensorFlow model
model.evaluate(validation_generator)

# Evaluation for PyTorch model
pytorch_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in dataloader:
        outputs = pytorch_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'PyTorch Model Accuracy: {accuracy}%')

# Save TensorFlow model
model.save('face_recognition_model_tf.h5')

# Save PyTorch model
torch.save(pytorch_model.state_dict(), 'face_recognition_model_pytorch.pth')
