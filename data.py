import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

def load_data(batch_size, data_path='./corruptmnist/'):
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.unsqueeze(0)),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_images = torch.load(data_path + 'train_images_0.pt')
    train_targets = torch.load(data_path + 'train_target_0.pt')

    # Apply the transform to each image in the dataset
    train_images = torch.stack([transform(img) for img in train_images])
    
    train_dataset = TensorDataset(train_images, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Assuming similar structure for test data
    test_images = torch.load(data_path + 'test_images.pt')  
    test_targets = torch.load(data_path + 'test_target.pt') 

    # Apply the transform to each image in the dataset
    test_images = torch.stack([transform(img) for img in test_images])

    test_dataset = TensorDataset(test_images, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

