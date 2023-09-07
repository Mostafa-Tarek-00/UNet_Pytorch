import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class carvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index]).replace(".jpg", "_mask.gif")
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask ==255]/= 255.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

            return image, mask
        
# import os
# from PIL import Image
# from torch.utils.data import Dataset
# import numpy as np

# class carvanaDataset(Dataset):
#     def __init__(self, image_dir, mask_dir, transform=None):
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.transform = transform
#         self.image_files = os.listdir(image_dir)

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, index):
#         img_name = self.image_files[index]
#         img_path = os.path.join(self.image_dir, img_name)

#         # Determine the mask file extension dynamically based on the image file extension
#         img_ext = img_name.split('.')[-1]
#         mask_ext = "png" if img_ext.lower() in ['jpg', 'jpeg', 'png'] else img_ext

#         mask_name = img_name.replace(f'.{img_ext}', f'.{mask_ext}')
#         mask_path = os.path.join(self.mask_dir, mask_name)

#         image = np.array(Image.open(img_path).convert("RGB"))
#         mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float64)

#         if self.transform is not None:
#             augmentations = self.transform(image=image, mask=mask)
#             image = augmentations["image"]
#             mask = augmentations["mask"]

#         return image, mask
