#Ideally we want our dataset code to be decoupled from our model training for better readability and modularity

#Import statements 
import torch #Core library for PyTorch
from torch.utils.data import Dataset #
from torchvision import datasets #Contains pre-defined datasets - like the FashionMNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt #Tool used for plotting and visualization

#tLoading the FashionMNIST Dataset from datasets.FashionMNIST
#Load training data
training_data = datasets.FashionMNIST(
    root="data", #Specific directory where the dataset will be stored or loaded from
    train=True, #Load the training
    download=True, #Download the dataset if it isnt already present
    transform=ToTensor() #Converts the images to PyTorch tensors
)

#Load testing data
test_data = datasets.FashionMNIST(
    root="data",
    train=False, #Test split
    download=True,
    transform=ToTensor()
)

#Label mapping that maps numeric labels (0-9( to the coresponding clothing item names in Dataset
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

#Visualizing sample imabes
figure = plt.figure(figsize=(8, 8)) #Creates a new figure for plotting with a size of 8x8inches
cols, rows = 3, 3 #defines the grid layout fore displaing images
for i in range(1, cols * rows + 1): #iterates to olot a rid of images
    sample_idx = torch.randint(len(training_data), size=(1,)).item() #Randomly selects and index
    img, label = training_data[sample_idx] #retrieves the image and label at chosen index
    figure.add_subplot(rows, cols, i) #Adds a subplot to the grid for each image
    plt.title(labels_map[label]) #sets title of subplot to clothing item name
    plt.axis("off") #Removes axis for cleaner look
    plt.imshow(img.squeeze(), cmap="gray") #displays image, removes any singleton diminsions, and renders the image in grayscale
plt.show() #Displays figure

#import statements
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader

#Must have 3 functions __init__, __len__, and __getitem__. 

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None): #run once when instantiating the Dataset object
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self): #Returns number of samples in our dataset
        return len(self.img_labels)

    def __getitem__(self, idx): #Loads and returns a sample from the dataset at the given index idx
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
