import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import random
from tqdm import tqdm

import utils



#Custome dataset for torch (for feature enconding)
class ImageDataset(Dataset):
    def __init__(self, img_data, label):
        self.img_data = img_data
        self.label = label
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def __len__(self):
        return len(self.img_data)
        
    def __getitem__(self, idx):
        img = cv2.imread(self.img_data[idx])
        
        #cover bgr to rgb
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #cover nparray to pil img
        img_pil = Image.fromarray(img_rgb)
        
        #transfer image to tensor
        img_tensor = self.transform(img_pil)
        
        if img_tensor.shape[1] != 224 or img_tensor.shape[2] != 224:
            print(f"Image at index {idx} ({self.img_data[idx]}) is not 224x224 after transform. Shape: {img_tensor.shape}")
        
        #transfer label to tensor
        label_tensor = torch.tensor(self.label)
        
        return img_tensor, label_tensor



#  etxtract the feature of the data
def feature_extractor(data, IDC_label):
    #create the dataset and the data loader for model
    dataset = ImageDataset(data, IDC_label)
    dataloader = DataLoader(dataset, batch_size = 32, shuffle = False, num_workers = 0)

    #loading pre-trained ResNet-50 model
    weight = models.ResNet50_Weights.DEFAULT
    resnet50 = models.resnet50(weights = weight)
    resnet50 = resnet50.to('cuda')
    resnet50.eval()# to evaluate model

    #Perform the feature extraciton on ResNet50
    #define the node to extract
    return_nodes = {
        'layer3' : 'encode_features',
    }
    #create extractor
    feature_extractor = create_feature_extractor(resnet50, return_nodes = return_nodes)

    all_features = []
    #feature extraction
    with torch.no_grad():
        for data, _ in tqdm(dataloader):
            data = data.to('cuda')
            features = feature_extractor(data)
            # avg_pool on layer3 and cover size to (N, 1024)
            pooled_features = F.adaptive_avg_pool2d(features['encode_features'], (1, 1)).view(features['encode_features'].size(0), -1)
            all_features.append(pooled_features)

    # combine all tensor in list into one
    img_features = torch.cat(all_features, dim=0).cpu()
    #img_features = torch.cat(all_features, dim=0)

    return img_features



# main function
def data_encoder(path):
    print("Loading the data.......")
    data, IDC_label, patients = utils.load_data(path)

    patient_data_map = utils.create_patient_data_map(patients)

    target_label_map = utils.create_target_label(patient_data_map, IDC_label)

    print("Encoding the data......")
    img_features = feature_extractor(data, IDC_label)
    print("Done")

    return img_features, patient_data_map, target_label_map




if __name__ == '__main__':
    # TESTING
    # data path
    path = "E:/gnn_project/datasets"
    #path = "/Volumes/SanDisk/gnn_project/datasets"

    # Data visualizaiton
    data, label, patients = utils.load_data(path)
    data = data[0:3000]
    IDC_label = label[0:3000]
    patients = patients[0:3000]
    patient_data_map = utils.create_patient_data_map(patients)

    # give target label
    target_label_map = utils.create_target_label(patient_data_map, IDC_label)
    target_count = {1: 0, 2: 0, 3: 0, 4: 0}
    for l in target_label_map.values():
        if l in target_count:
            target_count[l] += 1
    print(f"The target count: {target_count}")
    
    #explore the size of image
    temp_img = cv2.imread(data[30])
    plt.figure(figsize=(4,4))
    if temp_img is not None:
        print(f"size: {temp_img.shape}")
    plt.imshow(temp_img)
    plt.show()

    #plot some example of the images with their label
    plt.figure(figsize=(10,10))
    for i in range(10):
        index = random.randint(0, len(data))
        example_img = cv2.imread(data[index])
        plt.subplot(5,5,+1+i)
        plt.title("IDC"if label[index] == '1' else "non-IDC", fontsize = 12)
        plt.axis(False)
        plt.imshow(example_img)
    plt.show()

    # feature extraction
    img_features = feature_extractor(data, IDC_label)   

    print(img_features.size())
    print(img_features)

