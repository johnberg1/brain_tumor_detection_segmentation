from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utils import make_dataset

class ImageDataset(Dataset):
    
    def __init__(self, source_root, target_root):
        self.source_paths = sorted(make_dataset(source_root))
        self.target_paths = sorted(make_dataset(target_root))
        self.source_transforms = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((256,256))]) # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.target_transforms = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((256,256), interpolation=0)]) # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        
    def __len__(self):
        return len(self.source_paths)
    
    def __getitem__(self, index):
        from_path = self.source_paths[index]
        from_im = Image.open(from_path).convert('L')
        
        to_path = self.target_paths[index]
        to_im = Image.open(to_path).convert('L')
        
        from_im = self.source_transforms(from_im)
        to_im = self.target_transforms(to_im)
        return from_im, to_im
                                
            
        
        