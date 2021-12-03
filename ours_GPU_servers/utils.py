from torch.utils.data import Dataset
import numpy as np
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import math
import mat73
from skimage.transform import resize


from iou import IoU
from metrices import *
# from ours.Utils.iou import *   # Georgios
# from ours.Utils.metrices import *   # Georgios


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def pad_resize(img,desired_size):

    old_size = img.shape[:2]
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = resize(img, new_size, preserve_range=True)

    new_im = np.zeros((desired_size,desired_size,3))
    new_im[int((desired_size - new_size[0]) // 2):(int((desired_size - new_size[0]) // 2)+img.shape[0]),
    int((desired_size - new_size[1]) // 2):(int((desired_size - new_size[1]) // 2)+img.shape[1]),:] = img

    img_window = [int((desired_size - new_size[0]) // 2),(int((desired_size - new_size[0]) // 2)+img.shape[0]),
                  int((desired_size - new_size[1]) // 2), (int((desired_size - new_size[1]) // 2)+img.shape[1])]

    return Image.fromarray(np.uint8(new_im)).convert('RGB'), img_window


def min_max_normalize(image):
    image_min = np.min(image)
    image_max = np.max(image)

    image = (image-image_min) / (image_max-image_min)# + 1e-6) ## safe division

    return image


class ImageNetVal(Dataset):

    ### Overwriting some functions of Dataset build in class
    def __init__(self, img_folder, labels_dict, device, transform):

        self.labels_dict = np.load(labels_dict, allow_pickle=True)
        self.transform = transform
        self.img_names = os.listdir(img_folder)
        self.img_names = np.asarray([img_folder+"/"+current_img for current_img in self.img_names])
        self.device = device

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):


        current_path = self.img_names[idx]
        img_id = current_path.split("/")[-1]

        img_orig = Image.open(current_path)
        n_channels = img_orig.layers

        if n_channels == 1:
            img_orig = img_orig.convert(mode='RGB')

        img = self.transform(img_orig)

        label = torch.IntTensor([self.labels_dict[img_id]])


        return img.to(self.device), label.to(self.device)


class ImageNetSegm:
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

        # Specify transforms
        self.transform_img = transform_img
        self.transform_seg_mask = transform_seg_mask

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # The image with index idx
        img_orig = self.images[idx]
        if len(img_orig.shape) != 3:
            img_orig = img_orig.convert(mode='RGB')

        img_orig = Image.fromarray(img_orig)
        ## normalizing original image
        img_trans = self.transform_img(img_orig)

        ## resizing original image
        img_orig = self.transform_seg_mask(img_orig)

        ## resizing GT segmentation mask
        seg_mask_orig = self.seg_masks[idx]
        seg_mask_orig = Image.fromarray(seg_mask_orig[0])
        seg_mask_trans = self.transform_seg_mask(seg_mask_orig)


        return img_trans.to(self.device), seg_mask_trans.to(self.device), img_orig.to(self.device)



### Layer Initialization

def no_grad_trunc_normal_(tensor, mean=0, std=1., a=-2., b=2.):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


######### metrics

def eval_batch(Res, labels):
    ## stolen from https://github.com/hila-chefer/Transformer-Explainability
    ## Thanks Hila Chefer

    # threshold between FG and BG is the mean
    Res = (Res - Res.min()) / (Res.max() - Res.min())

    ret = Res.mean()

    Res_1 = Res.gt(ret).type(Res.type())
    Res_0 = Res.le(ret).type(Res.type())

    Res_1_AP = Res
    Res_0_AP = 1 - Res
    ######### ???????????????????????????????????
    Res_1[Res_1 != Res_1] = 0
    Res_0[Res_0 != Res_0] = 0
    Res_1_AP[Res_1_AP != Res_1_AP] = 0
    Res_0_AP[Res_0_AP != Res_0_AP] = 0

    # TEST
    pred = Res.clamp(0) / Res.max()
    pred = pred.view(-1).data.cpu().numpy()
    target = labels.view(-1).data.cpu().numpy()
    # print("target", target.shape)

    output = torch.cat((Res_0.unsqueeze(0), Res_1.unsqueeze(0)), 0)
    output_AP = torch.cat((Res_0_AP.unsqueeze(0), Res_1_AP.unsqueeze(0)), 0)

    # Evaluate Segmentation
    batch_inter, batch_union, batch_correct, batch_label, batch_ap = 0, 0, 0, 0, 0

    # Segmentation resutls
    correct, labeled = batch_pix_accuracy(output.data.cpu(), labels[0, 0])
    ## labeled: all positive pixels in the groundtruth
    ## correct: all positive pixels in the groundtruth that were also predicted as positive (larger than the mean)

    inter, union = batch_intersection_union(output.data.cpu(), labels[0, 0], 2)
    batch_correct += correct
    batch_label += labeled
    batch_inter += inter
    batch_union += union
    # print("output", output.shape)
    # print("ap labels", labels.shape)
    # ap = np.nan_to_num(get_ap_scores(output, labels))
    ap = np.nan_to_num(get_ap_scores(output_AP.unsqueeze(0), labels[0]))
    batch_ap += ap

    return batch_correct, batch_label, batch_inter, batch_union, batch_ap, pred, target


class PascalVOC2012(Dataset):

    ### Overwriting some functions of Dataset build in class
    def __init__(self, img_names, labels_dict, voc12_img_folder, input_dim, device, transform=None):

        self.labels_dict = np.load(labels_dict, allow_pickle=True).item()
        self.transform = transform
        self.input_dim = input_dim
        self.device = device


        with open(img_names) as file:
            self.img_paths = np.asarray([voc12_img_folder + l.rstrip("\n")+".jpg" for l in file])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        current_path = self.img_paths[idx]

        img_orig = plt.imread(current_path)
        img, window = pad_resize(img_orig, self.input_dim)

        img = self.transform(img)

        ### resizing and padding the image to fix dimensions inputs
        orginal_shape = np.shape(img_orig)

        img_key = current_path.split("/")[-1][:-4]
        label = torch.from_numpy(self.labels_dict[img_key])

        return current_path, img.to(self.device), label.to(self.device), window, orginal_shape
