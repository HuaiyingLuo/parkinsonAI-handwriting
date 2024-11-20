import os
from PIL import Image
from torch.utils.data import Dataset

class DualImageDataset(Dataset):
    def __init__(self, meander_root, spiral_root, transform=None):
        self.meander_root = meander_root
        self.spiral_root = spiral_root
        self.transform = transform

        self.classes = sorted([
            d for d in os.listdir(meander_root)
            if os.path.isdir(os.path.join(meander_root, d)) and not d.startswith('.')
        ])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        valid_image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

        self.samples = []
        for cls_name in self.classes:
            meander_cls_dir = os.path.join(meander_root, cls_name)
            spiral_cls_dir = os.path.join(spiral_root, cls_name)

            if not os.path.isdir(meander_cls_dir):
                print(f"Warning: Meander class directory {meander_cls_dir} does not exist.")
                continue
            if not os.path.isdir(spiral_cls_dir):
                print(f"Warning: Spiral class directory {spiral_cls_dir} does not exist.")
                continue

            meander_imgs = sorted([
                os.path.join(meander_cls_dir, img_name)
                for img_name in os.listdir(meander_cls_dir)
                if not img_name.startswith('.') and img_name.lower().endswith(valid_image_extensions)
            ])
            spiral_imgs = sorted([
                os.path.join(spiral_cls_dir, img_name)
                for img_name in os.listdir(spiral_cls_dir)
                if not img_name.startswith('.') and img_name.lower().endswith(valid_image_extensions)
            ])

            print(f"Class: {cls_name}, Meander images: {len(meander_imgs)}, Spiral images: {len(spiral_imgs)}")


            if len(meander_imgs) == 0 or len(spiral_imgs) == 0:
                print(f"Warning: No images found in class {cls_name}.")
                continue

            min_len = min(len(meander_imgs), len(spiral_imgs))

            for i in range(min_len):
                meander_img_path = meander_imgs[i]
                spiral_img_path = spiral_imgs[i]
                label = self.class_to_idx[cls_name]
                self.samples.append((meander_img_path, spiral_img_path, label))

        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meander_img_path, spiral_img_path, label = self.samples[idx]

        meander_image = Image.open(meander_img_path).convert('RGB')
        spiral_image = Image.open(spiral_img_path).convert('RGB')

        if self.transform:
            meander_image = self.transform(meander_image)
            spiral_image = self.transform(spiral_image)

        return meander_image, spiral_image, label
