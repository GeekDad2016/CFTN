from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class NarutoDataset(Dataset):
    def __init__(self, split='train', im_size=128):
        self.dataset = load_dataset("lambdalabs/naruto-blip-captions", split=split)
        self.transform = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        if image.mode == 'L': # handle grayscale images
            image = image.convert('RGB')
        
        # The text is also available in item['text'] if needed
        return self.transform(image), item['text']
