'''
Rasterize polygons to binary lake mask images, and remove images without matched polygons.

Author: Yi Wang

'''


import rasterio
import os
import numpy as np
from rasterio.features import rasterize
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from pyproj import CRS
import shapely
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--gpkg-path',type=str)
parser.add_argument('--patch-out',type=str)
parser.add_argument('--label-out',type=str)

args = parser.parse_args()

gpkg_path = args.gpkg_path
img_dir = os.path.join(args.patch_out,'train')
label_out = os.path.join(args.label_out,'train')
os.makedirs(label_out, exist_ok=True)

polygons = gpd.read_file(gpkg_path)
values_to_split = ["Greenland26X_22W_Sentinel2_2019-06-03_05.tif",
                   "Greenland26X_22W_Sentinel2_2019-06-19_20.tif",
                   "Greenland26X_22W_Sentinel2_2019-07-31_25.tif",
                   "Greenland26X_22W_Sentinel2_2019-08-25_29.tif"]
split_dataframes = {}
for value in values_to_split:
    split_dataframes[value] = polygons[polygons["image"] == value]


for img_name in tqdm(os.listdir(img_dir)):
    img_path = os.path.join(img_dir, img_name)
    if "0603" in img_name:
        polygons = split_dataframes["Greenland26X_22W_Sentinel2_2019-06-03_05.tif"]
    elif "0619" in img_name:
        polygons = split_dataframes["Greenland26X_22W_Sentinel2_2019-06-19_20.tif"]
    elif "0731" in img_name:
        polygons = split_dataframes["Greenland26X_22W_Sentinel2_2019-07-31_25.tif"]
    elif "0825" in img_name:
        polygons = split_dataframes["Greenland26X_22W_Sentinel2_2019-08-25_29.tif"]    
    else:
        print("ERROR: No polygons found")
        break

    with rasterio.open(img_path) as src:
        img = src.read()
        img_bbox = src.bounds
        crs = src.crs
        meta = src.meta

        # bbox to shapely geometry
        bbox = shapely.geometry.box(*img_bbox)
        # bbox to gdf
        bbox_gpd = gpd.GeoDataFrame({'geometry': [bbox]}, crs=CRS.from_wkt(polygons.crs.to_wkt()))
        #bbox_gpd = gpd.GeoDataFrame({'geometry': [bbox]}, crs=crs)
        # intersect for matched polygons
        gdf = gpd.overlay(polygons, bbox_gpd, how='intersection')

        if gdf.empty:
            # remove image if no matched polygons
            os.remove(img_path)
            continue
        
        # rasterize matched polygons to a mask image, 0 for background, 1 for foreground
        mask = rasterize(gdf.geometry, out_shape=img.shape[1:], transform=src.transform, fill=0, all_touched=True)
        
        # write mask image
        out_path = os.path.join(label_out, img_name)
        meta.update({
                    #'driver': 'GTiff',
                    #'height': mask.shape[0],
                    #'width': mask.shape[1],
                    #'transform': src.transform,
                    #'crs': src.crs,
                    'count': 1,
                    'dtype': rasterio.uint8})
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(mask.astype(rasterio.uint8), 1)
        


