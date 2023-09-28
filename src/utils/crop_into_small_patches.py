'''
Split the big images into 24 train/test regions, and then crop the big images into small patches.

Author: Chenying Liu

'''

import os
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.mask import mask
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--raw-dir',type=str)
parser.add_argument('--region-out',type=str)
parser.add_argument('--patch-out',type=str)

args = parser.parse_args()


def save_tif(output_path, data, crs, transform):
    # Create an empty raster to hold the rasterized features
    raster = rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs=crs,
        transform=transform
    )
    # # Write the rasterized data into the raster file
    raster.write(data)
    # Close the raster file
    raster.close()
    return



ps = 1024
train_region_ids = {'0603':[2,4,6],
                    '0619':[1,3,5],
                    '0731':[2,4,6],
                    '0825':[1,3,5]}
                   
test_region_ids = {'0603':[1,3,5],
                   '0619':[2,4,6],
                   '0731':[1,3,5],
                   '0825':[2,4,6]}

# image patch
ts_img_dir = args.raw_dir
fnames = [f for f in os.listdir(ts_img_dir) if f.endswith('.tif')]

# read region masks
region_mask_dir = os.path.join(args.raw_dir,'lakes_regions.gpkg') 
gdf_region = gpd.read_file(region_mask_dir) # columns: region_num, geometry
gdf_region.crs

# output directory
region_out_dir = args.region_out
if not os.path.exists(region_out_dir):
    os.makedirs(os.path.join(region_out_dir,'train'))
    os.makedirs(os.path.join(region_out_dir,'test'))
    
patch_out_dir = args.patch_out
if not os.path.exists(patch_out_dir):
    os.makedirs(os.path.join(patch_out_dir,'train'))
    os.makedirs(os.path.join(patch_out_dir,'test'))

#%% crop big images according to shapefile regions
for f in tqdm(fnames):
    dates = f.split('_')[-2].split('-')
    date_str = dates[1]+dates[2]
    for r in test_region_ids[date_str]:
        with rasterio.open(os.path.join(ts_img_dir, f)) as src:
            region_mask = gdf_region[gdf_region['region_num']==r]
            
            # Use the polygon to create a mask for cropping
            # mask_geom = [mapping(region_mask.geometry)]
            cropped_data, _ = mask(src, region_mask.geometry, crop=True)
            
            # Create a new output file path
            output_path = os.path.join(region_out_dir, 'test', f'{date_str}_r{r}.tif')

            # Update the cropped raster's metadata
            minx, maxy = region_mask.bounds.minx.values[0], region_mask.bounds.maxy.values[0]
            transform=rasterio.Affine(src.transform[0], 0, minx, 0, src.transform[4], maxy)

            # Save the cropped raster to disk
            save_tif(output_path, cropped_data, src.crs, transform)

    for r in train_region_ids[date_str]:
        with rasterio.open(os.path.join(ts_img_dir, f)) as src:
            region_mask = gdf_region[gdf_region['region_num']==r]
            
            # Use the polygon to create a mask for cropping
            # mask_geom = [mapping(region_mask.geometry)]
            cropped_data, _ = mask(src, region_mask.geometry, crop=True)
            
            # Create a new output file path
            output_path = os.path.join(region_out_dir, 'train', f'{date_str}_r{r}.tif')

            # Update the cropped raster's metadata
            minx, maxy = region_mask.bounds.minx.values[0], region_mask.bounds.maxy.values[0]
            transform=rasterio.Affine(src.transform[0], 0, minx, 0, src.transform[4], maxy)

            # Save the cropped raster to disk
            save_tif(output_path, cropped_data, src.crs, transform)


print("Images split into 24 train and test regions.")




region_out_dir_train = os.path.join(region_out_dir,'train')
region_out_dir_test = os.path.join(region_out_dir,'test')

#%% crop big images into small patches
fnames_r = [f for f in os.listdir(region_out_dir_train) if f.endswith('.tif')]
for f in tqdm(fnames_r):
    with rasterio.open(os.path.join(region_out_dir_train, f)) as src:
        img = src.read()
        crs = src.crs
        tr = src.transform
        xpixel_size, _, minx, _, ypixel_size, maxy = [src.transform[j] for j in range(6)]
    
    dim, height, width = img.shape
    nrow = np.ceil(height/ps).astype(int)
    ncol = np.ceil(width/ps).astype(int)
    
    # crop the big image into small patches
    for r in range(nrow):
        r_start = r*ps
        r_end = (r+1)*ps if r<(nrow-1) else height
        for c in range(ncol): 
            ind = r*ncol + c
            c_start = c*ps
            c_end = (c+1)*ps if c<(ncol-1) else width
            img_patch = img[:, r_start:r_end, c_start:c_end]
            
            # save patch
            fp_out = os.path.join(patch_out_dir, 'train', f'{f.split(".")[0]}.{ind}.tif')
            minx_sub = minx + c_start*xpixel_size
            maxy_sub = maxy + r_start*ypixel_size
            transform = rasterio.Affine(xpixel_size, 0, minx_sub, 0, ypixel_size, maxy_sub)
            
            # Save the cropped raster to disk
            save_tif(fp_out, img_patch, crs, transform)
               
fnames_r = [f for f in os.listdir(region_out_dir_test) if f.endswith('.tif')]
for f in tqdm(fnames_r):
    with rasterio.open(os.path.join(region_out_dir_test, f)) as src:
        img = src.read()
        crs = src.crs
        tr = src.transform
        xpixel_size, _, minx, _, ypixel_size, maxy = [src.transform[j] for j in range(6)]
    
    dim, height, width = img.shape
    nrow = np.ceil(height/ps).astype(int)
    ncol = np.ceil(width/ps).astype(int)
    
    # crop the big image into small patches
    for r in range(nrow):
        r_start = r*ps
        r_end = (r+1)*ps if r<(nrow-1) else height
        for c in range(ncol): 
            ind = r*ncol + c
            c_start = c*ps
            c_end = (c+1)*ps if c<(ncol-1) else width
            img_patch = img[:, r_start:r_end, c_start:c_end]
            
            # save patch
            fp_out = os.path.join(patch_out_dir, 'test', f'{f.split(".")[0]}.{ind}.tif')
            minx_sub = minx + c_start*xpixel_size
            maxy_sub = maxy + r_start*ypixel_size
            transform = rasterio.Affine(xpixel_size, 0, minx_sub, 0, ypixel_size, maxy_sub)
            
            # Save the cropped raster to disk
            save_tif(fp_out, img_patch, crs, transform)            
            
            
print("Images cropped into 1024x1024 patches.")