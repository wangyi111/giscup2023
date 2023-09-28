# GISCUP2023

This repository contains the code for the [ACM SIGSPATIAL GISCUP 2023](https://sigspatial2023.sigspatial.org/giscup/index.html) competition.

## Usage
```
$ cd src
```

### Data preparation

Download the Sentinel-2 images, the `lake_region.gpkg`, and the `lake_polygons_training.gpkg` from the [challenge website](https://sigspatial2023.sigspatial.org/giscup/download.html). Create a folder `data` and put the files in it.

Split regions and crop the big images into small patches:
```
python utils/crop_into_small_patches.py --raw-dir data/ --region-out data/preprocess/img_region_out/ --patch-out data/preprocess/img_patch_out/
```

Rasterize labels to binary masks and filter images without valid polygons:
```
python utils/rasterize_polygons.py --gpkg-path data/lake_polygons_training.gpkg --patch-out data/preprocess/img_patch_out/ --label-out data/preprocess/label_patch_out
```

### Training

We train [mask2former](https://arxiv.org/abs/2112.01527) as a binary segmentation task, using the [mmsegmentation toolbox](https://github.com/open-mmlab/mmsegmentation). Refer to the official repo for installation and dependencies.

Before training, organize the data into the format of mmsegmentation:
```
mmsegmentation
├── mmseg
├── tools
├── configs
├── data
│   ├── ice_lake
│   │   ├── train
│   │   │   ├── images
│   │   │   ├── semantics
│   │   ├── test
│   │   │   ├── images
│   │   │   ├── semantics
```

Then, train the model:
```
python tools/train.py configs/mask2former/mask2former_swin-b-in22k-384x384-pre_4xb2-15k_ice_lake-1024x1024.py work_dirs/mask2former_swin-b-in22k-384x384-pre_4xb2-15k_ice_lake-1024x1024
```

We trained the model with DistributedDataParallel on a node with 4 GPUs:
```
bash tools/dist_train.sh configs/mask2former/mask2former_swin-b-in22k-384x384-pre_4xb2-15k_ice_lake-1024x1024.py work_dirs/mask2former_swin-b-in22k-384x384-pre_4xb2-15k_ice_lake-1024x1024 4
```

### Inference

Inference on the test set with test-time augmentation:
```
bash tools/dist_test.sh configs/mask2former/mask2former_swin-b-in22k-384x384-pre_4xb2-15k_ice_lake-1024x1024.py work_dirs/mask2former_swin-b-in22k-384x384-pre_4xb2-15k_ice_lake-1024x1024/epoch_15.pth 4 --tta
```

This will generate the predicted test masks in `work_dirs/pred_ice_lake_mask2former_swinb12_it15k_test_tta`.

### Post-processing

Fill the holes in the predicted lakes, aggregate the small patches into region-based big masks:
```
python utils/aggregate_seg_predict_patches.py --img-dir data/preprocess/img_patch_out/test/ --pred-dir mmsegmentation/work_dirs/pred_ice_lake_mask2former_swinb12_it15k_test_tta/ --save-dir data/postprocess/region_pred
```

This will generate 12 region-based test masks in `data/postprocess/region_pred`.

Finally, add geo-coordinates to the region-based masks, convert to polygons, filter out small outliers, and save to `data/postprocess/lake_polygons_test.gpkg`:
```
python utils/png2tif2gpkg.py --img-dir data/preprocess/img_region_out/test/ --pred-dir data/postprocess/region_pred/ --save-path data/postprocess/lake_polygons_test.gpkg --train-path data/lake_polygons_training.gpkg
```

