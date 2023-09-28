'''
Postprocess and aggregate segmentation results from patches to region-based big images.

Author: Chenying Liu

'''

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from tqdm import tqdm
import argparse
import pdb

parser = argparse.ArgumentParser()

parser.add_argument('--img-dir',type=str)
parser.add_argument('--pred-dir',type=str)
parser.add_argument('--save-dir',type=str)

args = parser.parse_args()


def find_instance_and_fill_holes(seg):
    # find instance contours from seg
    cnts, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # index objects
    mask = np.zeros(list(seg.shape)+[3], dtype=np.uint8)
    for ci, cnt in enumerate(cnts):
        cv2.drawContours(mask, [cnt], 0, (ci+1,ci+1,ci+1), thickness=cv2.FILLED)
        # areas.append(np.sum(mask[:,:,0]==ci+1))

    # convert from RGB to gray
    mask = mask[:,:,0] # cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # mask = mask*seg  # to refine the shapes of objects in index file

    return mask


# original image patches
img_dir = args.img_dir
fnps = os.listdir(img_dir)
uni_patches = np.unique(np.array([fnp[:7] for fnp in fnps])).tolist()
patch_splits = {}
for up in uni_patches:
    patch_splits[up] = [fnp[:-4] for fnp in fnps if fnp[:7]==up]

print(uni_patches)

# segmentation results directory
pred_dir = args.pred_dir
fnames = os.listdir(pred_dir)

# save directory
save_dir = args.save_dir

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

#pdb.set_trace()

# stitch patches
preds = {}
for k in tqdm(uni_patches):
    patches = patch_splits[k]
    ps = 1024
    hs = []
    ws = []
    for i in range(len(patches)):
        fp = os.path.join(img_dir,k+f'.{i}.tif')
        with rasterio.open(fp,'r') as rd:
            hs.append(rd.height)
            ws.append(rd.width)

    nh = np.sum(np.array(ws)!=ps)
    nw = np.sum(np.array(hs)!=ps)
    h_less = np.array(hs)[np.array(hs)!=ps][0]
    w_less = np.array(ws)[np.array(ws)!=ps][0]

    pred = np.zeros(((nh-1)*ps+h_less,(nw-1)*ps+w_less))
    for h in range(nh):
        tmp = []
        for w in range(nw):
            ind = h*nw+w
            fp = os.path.join(pred_dir,k+f'.{ind}.png')
            if os.path.exists(fp):
                seg = cv2.imread(fp, 0)
                sh, sw = seg.shape
                pred[h*ps:h*ps+sh,w*ps:w*ps+sw] = seg
    preds[k] = pred

#pdb.set_trace()

# find instances and fill holes
ins_all = {}
for k in preds:
    print(k)
    pred = preds[k]
    ins = find_instance_and_fill_holes(pred)
    
    ins_all[k] = ins
    cv2.imwrite(os.path.join(save_dir,k+'.png'), ins)