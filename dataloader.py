import os
from PIL import Image
from torch.utils.data import Dataset

class TextImageDataset(Dataset):
    def __init__(self, text_file, image_dir, transform=None):
        self.text_file = text_file
        self.image_dir = image_dir
        self.transform = transform
        self.data = []
        with open(self.text_file, 'r', encoding='utf-8') as f:
            info = f.readlines()[1:]
            for line in info:
                line_info = line.strip().split(',')
                image_file = line_info[0]
                text = ','.join(line_info[1:]).strip()
                self.data.append((image_file, text))
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_file, text = self.data[index]
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, text

