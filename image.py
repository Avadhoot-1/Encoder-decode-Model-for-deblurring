from torchsummary import summary
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision.models as models
import pickle
from tqdm import tqdm
import numpy as np
from efficientnet_pytorch import EfficientNet

#=====================================================================================================================================#
# np.random.seed(0)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available! Using GPU for computations.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU for computations.")

#=====================================================================================================================================#

#######          DENSENET MODEL         #############
model_dense = models.densenet121(pretrained= True)

for param in model_dense.parameters():
    param.requires_grad = False

num_features_1 = model_dense.classifier.in_features
model_dense.classifier = nn.Sequential(nn.Linear(num_features_1, 200), nn.Softmax(dim = 1))

#=====================================================================================================================================#
#######          Efficient_B0         #############
model_eff = models.efficientnet_b0(models.EfficientNet_B0_Weights)
for param in model_eff.parameters():
    param.requires_grad = False

num_features_1 = model_eff.classifier[1].in_features
model_eff.classifier[1] = nn.Sequential(nn.Linear(num_features_1, 200), nn.Softmax(dim = 1))

#=====================================================================================================================================#
#########       Shufflenet              ##############

model_shuff = models.shufflenet_v2_x1_5(models.ShuffleNet_V2_X1_5_Weights)
for param in model_shuff.parameters():
    param.requires_grad = False

num_features_1 = model_shuff.fc.in_features
model_shuff.fc = nn.Sequential(nn.Linear(num_features_1, 200), nn.Softmax(dim = 1))

#=====================================================================================================================================#
########         Concatenated Model     ############
class ConcatenateModel(nn.Module):
    def __init__(self, model_eff, model_shuff):
        super(ConcatenateModel, self).__init__()
        self.mod1 = model_eff
        self.mod2 = model_shuff

    def forward(self, x):
       x1 = self.mod1(x.clone())
       x2 = self.mod2(x)
       output = (x1+x2)/2
       return output

model_conc = ConcatenateModel(model_eff, model_shuff)

model_map = {'shufflenet-v2-x1-5': model_shuff, 'efficientnet-b0': model_eff, 'densenet121': model_dense, 'shufflenet+efficient': model_conc}

class CustomDataset(Dataset):
    def __init__(self, data_file, labels):
        self.data = torch.load(data_file)  # Load data from .pt file
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        # Get data and label for the given index
        data_item = self.data[idx]
        label = self.labels[idx]
        return data_item, label

#==========================================================================================================================================
#==========================================================================================================================================
#==========================================================================================================================================

def Training(type, batch_sz, aug):
    torch.cuda.empty_cache()
    model_name = type + "_" + str(batch_sz) + "_" + str(aug)
    model = model_map[type]
    total_params = sum(p.numel() for p in model.parameters())
    model.to(device)

    if(aug):
        with open("train_label_aug.pkl", "rb") as file:
            y_train_old = pickle.load(file)

        with open("test_label_aug.pkl", "rb") as file:
            y_test_old = pickle.load(file)
    else:
        with open("train_label.pkl", "rb") as file:
            y_train_old = pickle.load(file)

        with open("test_label.pkl", "rb") as file:
            y_test_old = pickle.load(file)        

    y_train = [torch.tensor(int(item)-1).to(device) for item in y_train_old]
    y_test = [torch.tensor(int(item)-1).to(device) for item in y_test_old]
    if(aug):
        dataset = CustomDataset('preprocessed_images_train_aug.pt', y_train)
        dataset_test = CustomDataset('preprocessed_images_test_aug.pt', y_test)
    else:
        dataset = CustomDataset('preprocessed_images_train.pt', y_train)
        dataset_test = CustomDataset('preprocessed_images_test.pt', y_test)        

    batch_size = batch_sz
    shuffle = True 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    testloader = DataLoader(dataset_test, batch_size=32)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Train the model or load your own data for fine-tuning

    print("Training...")
    loss_list = []
    acuraccy_list = []
    for epoch in range(1,101):
        training_loss = 0
        correct = 0
        total = 0
        model.train()
        for inputs, labels in tqdm(dataloader, desc="Processing"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            training_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Calculate accuracy at the end of each epoch
        if(epoch%10 == 0):
            torch.save({
            'model_state_dict': model.state_dict(),
            }, f'model_checkpoint_concat{epoch}.pth')
        if(epoch%10 == 0):
            correct = 0
            total = 0
            for inputs, labels in tqdm(testloader, desc="Processing"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                max_val, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f'accuracy : {100*correct/total:.2f}')
            acuraccy_list.append(100*correct/total)
            
        loss_list.append(training_loss / len(dataloader))
        print(f'Epoch [{epoch + 1}/100], Loss: {training_loss / len(dataloader):.4f}')
    
    model_dict = {'name': model_name, 'params': total_params, 'loss_list': loss_list, 'acc': acuraccy_list}
    
    with open(f'data/{model_name}.pkl', "wb") as file:
        pickle.dump(model_dict, file) 



import argparse
 
def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Process some arguments.')
 
    # Add arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--model_name', type=str, default='resnet50', help='Name of the model')
    parser.add_argument('--augmented', action='store_true', help='Whether to use augmented data')
 
    # Parse the arguments
    args = parser.parse_args()
 
    # Access the arguments
    batch_size = args.batch_size
    model_name = args.model_name
    augmented = args.augmented
 
    # Print the arguments
    print(f'Batch Size: {batch_size}')
    print(f'Model Name: {model_name}')
    print(f'Augmented: {augmented}')
    
    Training(model_name, batch_size, augmented)
 
if __name__ == "__main__":
    main()

