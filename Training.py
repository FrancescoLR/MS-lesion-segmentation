"""
@author: Francesco La Rosa
"""

import os
import tempfile
import matplotlib.pyplot as plt
from glob import glob
import torch
import re
from torch import nn
from monai.data import CacheDataset, DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, GeneralizedDiceLoss, TverskyLoss
from monai.metrics import compute_meandice
from monai.networks.nets import UNet
from monai.transforms import (
    AddChanneld,Compose,CropForegroundd,LoadNiftid,Orientationd,RandCropByPosNegLabeld,
    ScaleIntensityRanged,Spacingd,ToTensord,ConcatItemsd,NormalizeIntensityd, RandFlipd,
    RandRotate90d,RandShiftIntensityd,RandAffined,RandSpatialCropd, RandScaleIntensityd, Activations,SqueezeDimd)
from monai.utils import first
import numpy as np


def main(temp):
    
    root_dir= ''  # Path where the trained model is located
    path_data = ''  # Path where the data is
    flair = sorted(glob(os.path.join(path_data, "*/FLAIR.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))
    mp2rage = sorted(glob(os.path.join(path_data, "*/MP2RAGE.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))
    segs = sorted(glob(os.path.join(path_data, "*/gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i)))

    N = (len(flair)) # Number of subjects for training/validation, by default using all subjects in the folder
    
    np.random.seed(111)
    indices = np.random.permutation(N)
    # 5 random cases are kept for validation, the others for training
    v=indices[:5]
    t=indices[5:]

    train_files=[]
    val_files=[]
    for i in t:
        train_files = train_files + [{"flair": fl,"mp2rage": mp2,"label": seg} for fl, mp2, seg in zip(flair[i:i+1], mp2rage[i:i+1], segs[i:i+1])]
    for j in v:
        val_files = val_files + [{"flair": fl,"mprage": mp,"label": seg} for fl, mp, seg in zip(flair[j:j+1], mprage[j:j+1], segs[j:j+1])]
    print("Training cases:", len(train_files))
    print("Validation cases:", len(val_files))
    
    train_transforms = Compose(
    [
        LoadNiftid(keys=["flair", "mprage","label"]),
        
        AddChanneld(keys=["flair", "mprage","label"]),
        Spacingd(keys=["flair","mprage","label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear","nearest")),
        NormalizeIntensityd(keys=["flair", "mprage"], nonzero=True),
        RandShiftIntensityd(keys="flair",offsets=0.1,prob=1.0),
        RandShiftIntensityd(keys="mprage",offsets=0.1,prob=1.0),
        RandScaleIntensityd(keys="flair",factors=0.1,prob=1.0),
        RandScaleIntensityd(keys="mprage",factors=0.1,prob=1.0),
        ConcatItemsd(keys=["flair", "mprage"], name="image"),
        RandCropByPosNegLabeld(keys=["image", "label"],label_key="label",spatial_size=(128, 128, 128),
            pos=4,neg=1,num_samples=32,image_key="image"),
        RandSpatialCropd(keys=["image", "label"], roi_size=(96,96,96), random_center=True, random_size=False),
        RandFlipd (keys=["image", "label"],prob=0.5,spatial_axis=(0,1,2)),
        RandRotate90d (keys=["image", "label"],prob=0.5,spatial_axes=(0,1)),
        RandRotate90d (keys=["image", "label"],prob=0.5,spatial_axes=(1,2)),
        RandRotate90d (keys=["image", "label"],prob=0.5,spatial_axes=(0,2)),
        RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=1.0, spatial_size=(96, 96, 96),
                     rotate_range=(np.pi/12, np.pi/12, np.pi/12), scale_range=(0.1, 0.1, 0.1),padding_mode='border'),
        ToTensord(keys=["image", "label"]),
    ]
    )
    val_transforms = Compose(
    [
        LoadNiftid(keys=["flair", "mprage", "label"]),
        
        AddChanneld(keys=["flair", "mprage","label"]),
        Spacingd(keys=["flair", "mprage","label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear","nearest")),
        NormalizeIntensityd(keys=["flair", "mprage"], nonzero=True),
        ConcatItemsd(keys=["flair", "mprage"], name="image"),
        ToTensord(keys=["image", "label"]),
    ]
    )
    
    #%% Print an example slice with segmentation
    
    check_ds = Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    image, label = (check_data["image"][0][0], check_data["label"][0][0])
    mp2rage = check_data["image"][0][1]
    plt.figure("check", (12, 6))
    plt.subplot(1, 3, 1)
    plt.title("flair")
    plt.imshow(image[:, :, 48], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("mp2rage")
    plt.imshow(mp2rage[:, :, 48], cmap="gray")
    plt.subplot(1, 3, 3)
    plt.title("label")
    plt.imshow(label[:, :, 48])
    plt.show()

    #%%
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)

    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.5, num_workers=0)
    val_train_ds = CacheDataset(data=train_files, transform=val_transforms, cache_rate=0.5, num_workers=0)

    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)
    val_train_loader = DataLoader(val_train_ds, batch_size=1, num_workers=0)
  
    device = torch.device("cuda:0")
    model = UNet(
    dimensions=3,
    in_channels=2,
    out_channels=2,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    num_res_units=0).to(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True, sigmoid=False,
                             include_background=False)

    optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    model.load_state_dict(torch.load(os.path.join(root_dir, "Initial_model.pth"))) # Load a model, only if fine-tuning.
    
    epoch_num = 200
    val_interval = 5
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    metric_values_train = list()
    act = Activations(softmax=True)
    thresh = 0.4

    for epoch in range(epoch_num):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:

            n_samples = batch_data["image"].size(0)
            for m in range(0,batch_data["image"].size(0),2):
                step += 2
                inputs, labels = (
                    batch_data["image"][m:(m+2)].to(device),
                    batch_data["label"][m:(m+2)].type(torch.LongTensor).to(device))
                optimizer.zero_grad()
                outputs = model(inputs)
                
                #Dice loss
                loss1 = loss_function(outputs,labels)
                
                #Focal loss
                ce_loss = nn.CrossEntropyLoss(reduction='none')
                ce = ce_loss((outputs),torch.squeeze(labels,dim=1))
                gamma = 2.0
                pt = torch.exp(-ce)
                f_loss = 1*(1-pt)**gamma * ce 
                loss2=f_loss
                loss2 = torch.mean(loss2)
                loss = 0.5*loss1+loss2              
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                if step%100 == 0:
                    step_print = int(step/2)
                    print(f"{step_print}/{(len(train_ds)*n_samples) // (train_loader.batch_size*2)}, train_loss: {loss.item():.4f}")

        epoch_loss /= step_print
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                metric_sum = 0.0
                metric_count = 0
                for val_data in val_loader:
                    val_inputs, val_labels = (
                                val_data["image"].to(device),
                                val_data["label"].to(device),
                                )
                    roi_size = (96, 96, 96)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model,mode='gaussian')
                   
                    val_labels = val_labels.cpu().numpy()
                    gt = np.squeeze(val_labels)
                    val_outputs = act(val_outputs).cpu().numpy()
                    seg= np.squeeze(val_outputs[0,1])
                    seg[seg>thresh]=1
                    seg[seg<thresh]=0
                    value = (np.sum(seg[gt==1])*2.0) / (np.sum(seg) + np.sum(gt))

                    metric_count += 1
                    metric_sum += value.sum().item()
                metric = metric_sum / metric_count
                metric_values.append(metric)
                metric_sum_train = 0.0
                metric_count_train = 0
                for train_data in val_train_loader:
                    train_inputs, train_labels = (
                                train_data["image"].to(device),
                                train_data["label"].to(device),
                                )
                    roi_size = (96, 96, 96)
                    sw_batch_size = 4
                    train_outputs = sliding_window_inference(train_inputs, roi_size, sw_batch_size, model,mode='gaussian')
                    
                    train_labels = train_labels.cpu().numpy()
                    gt = np.squeeze(train_labels)
                    train_outputs = act(train_outputs).cpu().numpy()
                    seg= np.squeeze(train_outputs[0,1])
                    seg[seg>thresh]=1
                    seg[seg<thresh]=0
                    value_train = (np.sum(seg[gt==1])*2.0) / (np.sum(seg) + np.sum(gt))
                    
                    metric_count_train += 1
                    metric_sum_train += value_train.sum().item()    
                metric_train = metric_sum_train / metric_count_train
                metric_values_train.append(metric_train)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(root_dir, "Best_model.pth"))
                    print("saved new best metric model")
                print(f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                                    f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
                                    )
                plt.figure("train", (12, 6))
                plt.subplot(1, 2, 1)
                plt.title("Epoch Average Train Loss")
                x = [i + 1 for i in range(len(epoch_loss_values))]
                y = epoch_loss_values
                plt.xlabel("epoch")
                plt.plot(x, y)
                plt.subplot(1, 2, 2)
                plt.title("Val and Train Mean Dice")
                x = [val_interval * (i + 1) for i in range(len(metric_values))]
                y = metric_values
                y1 = metric_values_train
                plt.xlabel("epoch")
                plt.plot(x, y)
                plt.plot(x, y1)
                plt.show()     
          
#%%
if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        main(tempdir)