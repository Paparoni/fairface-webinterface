import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# Define the transform to be applied to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define the dataset class
class FaceDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_paths = os.listdir(folder_path)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_path, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        age, sex, race = self.image_paths[idx].split('_')
        age = int(age)
        sex = int(sex)
        race = int(race)
        return image, age, sex, race

# Define the neural network architecture
class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Define the dataset and dataloader
dataset = FaceDataset('sample_faces/')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the model and optimizer
model = FaceNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for batch_idx, (data, age, sex, race) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        age_loss = criterion(output[:, 0], age)
        sex_loss = criterion(output[:, 1], sex)
        race_loss = criterion(output[:, 2], race)
        loss = age_loss + sex_loss + race_loss
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx+1}/{len(dataloader)}, Loss: {loss.item()}')
torch.save(model.state_dict(), 'face_detection_model.pt')

