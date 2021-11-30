import mat73
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plot
from types import SimpleNamespace
from PIL import Image

args = SimpleNamespace(batch_size=1,
                       input_dim=224,
                       pretrained_weights="pretrained/vgg16_20M.pth",
                       val_set="ILSVRC2012_img_val",
                       labels_dict="val_labels_dict.npy",
                       device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                       )

class Loader:
    def __init__(self, path, device, transform_img, transform_seg_mask):
        # Load the data
        data = mat73.loadmat(path)
        data = data['value']

        # Store the device
        self.device = device

        # Extract arguments
        self.n_images = int(data['n'].item())
        self.images = data['img']
        self.image_ids = data['id']
        self.seg_masks = data['gt']
        self.targets = data['target']

        # Specify transforms
        self.transform_img = transform_img
        self.transform_seg_mask = transform_seg_mask

        # Determine IDs
        class_labels = []
        for label in self.targets:
            if(class_labels.count(label) == 0):
                class_labels.append(label)

        i = 0
        self.labels_idx = {}
        self.labels_map = {}
        for label in class_labels:
            self.labels_idx[label] = i
            self.labels_map[i] = label
            i += 1

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # The image with index idx
        img_orig = self.images[idx]
        n_channels = img_orig.shape[-1]     # todo --> check
        if n_channels == 1:
            img_orig = img_orig.convert(mode='RGB')

        img_pil = Image.fromarray(img_orig)
        img_trans = self.transform_img(img_pil)

        # The ground truth segmentation mask
        seg_mask_orig = self.seg_masks[idx]
        seg_mask_pil = Image.fromarray(seg_mask_orig[0])
        seg_mask_trans = self.transform_seg_mask(seg_mask_pil)

        return img_trans.to(self.device), seg_mask_trans.to(self.device)

path = r'C:\Users\georg\Documents\KTH_ML_Master\Deep Learning Advanced Course\Project\Datasets\gtsegs_ijcv.mat'

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
transform = transforms.Compose([
    transforms.Resize((args.input_dim, args.input_dim)),
    transforms.ToTensor(),
    normalize,
])

transform_gt_mask = transforms.Compose([
    transforms.Resize(args.input_dim),
    transforms.ToTensor(),
])

dataLoader = Loader(path, args.device, transform, transform_gt_mask)


print("DONE")