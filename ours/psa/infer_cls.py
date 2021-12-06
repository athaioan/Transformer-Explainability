
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
import voc12.data
import scipy.misc
import importlib
from torch.utils.data import DataLoader
import torchvision
from tool import imutils, pyutils
import argparse
from PIL import Image
import torch.nn.functional as F
import os.path
import matplotlib.pyplot as plt
import imageio

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=r"C:\Users\johny\Desktop\Transformer-Explainability-main\psa-master\voc12\vgg_cls.pth", type=str),
    parser.add_argument("--network", default="network.vgg16_cls", type=str),
    parser.add_argument("--infer_list", default=r"C:\Users\johny\Desktop\Transformer-Explainability-main\psa-master\voc12\val.txt", type=str)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--voc12_root", default = r"C:\Users\johny\Desktop\Transformer-Explainability-main\ours\VOCdevkit\VOC2012", required=False, type=str)
    parser.add_argument("--low_alpha", default=4, type=int)
    parser.add_argument("--high_alpha", default=32, type=int)
    parser.add_argument("--out_cam", default=r"C:\Users\johny\Desktop\Transformer-Explainability-main\psa-master\out_cam", type=str)
    parser.add_argument("--out_la_crf", default=r"C:\Users\johny\Desktop\Transformer-Explainability-main\psa-master\out_la_crf", type=str)
    parser.add_argument("--out_ha_crf", default=r"C:\Users\johny\Desktop\Transformer-Explainability-main\psa-master\out_ha_crf", type=str)
    parser.add_argument("--out_cam_pred", default=r"C:\Users\johny\Desktop\Transformer-Explainability-main\psa-master\out_cam_pred", type=str)

    args = parser.parse_args()

    model = getattr(importlib.import_module(args.network), 'Net')()
    model.load_state_dict(torch.load(args.weights))

    model.eval()
    model.cuda()

    infer_dataset = voc12.data.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root,
                                                   scales=(1, 0.5, 1.5, 2.0),
                                                   inter_transform=torchvision.transforms.Compose(
                                                       [np.asarray,
                                                        model.normalize,
                                                        imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))

    for iter, (img_name, img_list, label) in enumerate(infer_data_loader):
        img_name = img_name[0]; label = label[0]

        img_path = voc12.data.get_img_path(img_name, args.voc12_root)
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]

        def _work(i, img):
            with torch.no_grad():
                with torch.cuda.device(i%n_gpus):
                    cam = model_replicas[i%n_gpus].forward_cam(img.cuda())
                    cam = F.upsample(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
                    cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
                    if i % 2 == 1:
                        cam = np.flip(cam, axis=-1)
                    return cam


        thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list)),
                                            batch_size=12, prefetch_size=0, processes=args.num_workers)

        cam_list = thread_pool.pop_results()
        # cam_list = [np.asarray(cam_list)]

        sum_cam = np.sum(cam_list, axis=0)
        norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5)

        cam_dict = {}
        for i in range(20):
            if label[i] > 1e-5:
                cam_dict[i] = norm_cam[i]

        if args.out_cam is not None:
            np.save(args.out_cam +"/"+ img_name + '.npy', cam_dict)

        if args.out_cam_pred is not None:
            bg_score = [np.ones_like(norm_cam[0])*0.2]
            pred = np.argmax(np.concatenate((bg_score, norm_cam)), 0)
            imageio.imwrite(args.out_cam_pred +"/"+  img_name + '.png', pred.astype(np.uint8))

        def _crf_with_alpha(cam_dict, alpha):
            v = np.array(list(cam_dict.values()))
            bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
            bgcam_score = np.concatenate((bg_score, v), axis=0)
            crf_score = imutils.crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

            n_crf_al = dict()

            n_crf_al[0] = crf_score[0]
            for i, key in enumerate(cam_dict.keys()):
                n_crf_al[key+1] = crf_score[i+1]

            return n_crf_al

        if args.out_la_crf is not None:
            crf_la = _crf_with_alpha(cam_dict, args.low_alpha)
            np.save(os.path.join(args.out_la_crf, img_name + '.npy'), crf_la)

        if args.out_ha_crf is not None:
            crf_ha = _crf_with_alpha(cam_dict, args.high_alpha)
            np.save(os.path.join(args.out_ha_crf, img_name + '.npy'), crf_ha)

        print(iter)

    def _fast_hist(label_true, label_pred, n_class):

        # source https://github.com/Juliachang/SC-CAM
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)

        return hist


    def scores(label_trues, label_preds, n_class):
        # https://github.com/Juliachang/SC-CAM

        hist = np.zeros((n_class, n_class))

        for lt, lp in zip(label_trues, label_preds):
            hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        valid = hist.sum(axis=1) > 0  # added
        mean_iu = np.nanmean(iu[valid])
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(n_class), iu))

        return {
            "Pixel Accuracy": acc,
            "Mean Accuracy": acc_cls,
            "Frequency Weighted IoU": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
        }


    # import os
    gt_mask_fold = "C:/Users/johny/Desktop/Transformer-Explainability-main/ours/VOCdevkit/VOC2012/SegmentationClass/"
    cam_fold = "C:/Users/johny/Desktop/Transformer-Explainability-main/psa-master/out_cam_pred/"
    cam_fold = "C:/Users/johny/Desktop/Transformer-Explainability-main/psa-master/out_rw/"

    cams = os.listdir(cam_fold)

    label_trues = []
    label_preds = []
    for index, current_cam in enumerate(cams):
        print("Step",index/len(cams))
        current_cam_path = cam_fold + current_cam
        current_mask_path = gt_mask_fold + current_cam

        cam_output = Image.open(cam_fold + current_cam)
        cam_output = np.array(cam_output)

        ## loading ground truth annotated mask
        gt_mask = Image.open(current_mask_path)
        gt_mask = np.array(gt_mask)

        label_preds.append(cam_output)
        label_trues.append(gt_mask)

    metrics = scores(label_trues, label_preds, 21)
    print("")