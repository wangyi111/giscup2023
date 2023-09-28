'''
prediction png to geotiff, then polygonize to geopackage.

Author: Yao Sun

'''


import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import shape  # Import the 'shape' function
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img-dir',type=str)
parser.add_argument('--pred-dir',type=str)
parser.add_argument('--save-path',type=str)
parser.add_argument('--train-path',type=str)
args = parser.parse_args()


def replace_imname(imname_input):
    ''' to add the image name filed in vector data: replace image patch name with original image name.
    Parameters
    ----------
    imname_input : TYPE
        first part of image patch name, indicating the original image name and region.

    Returns
    -------
    im : TYPE
        original image name, put in the filed of vector data.

    '''
    if imname_input == '0603':
        im = 'Greenland26X_22W_Sentinel2_2019-06-03_05.tif'
    elif imname_input == '0619':
        im = 'Greenland26X_22W_Sentinel2_2019-06-19_20.tif'
    elif imname_input == '0731':
        im = 'Greenland26X_22W_Sentinel2_2019-07-31_25.tif'
    elif imname_input == '0825':
        im = 'Greenland26X_22W_Sentinel2_2019-08-25_29.tif'
            
    return im

def png2tif_w_ref_allin1folder(png_folder, tiff_folder):
    ''' for all pngs in png_folder, add georeference according to the corresponding tif in tiff_folder, save mask-tif in png_folder.
    '''
    
    # List all TIFF files in the TIFF folder
    png_files = [f for f in os.listdir(png_folder) if f.endswith(".png")]

    # Iterate through the TIFF files and check if there is a matching PNG file
    for png_file in tqdm(png_files):
        # Extract the base file name without the extension
        base_name = os.path.splitext(png_file)[0]
    
        reference_geotiff_path = os.path.join(tiff_folder, base_name + ".tif")
        save_geotiff_path = os.path.join(png_folder, base_name + ".tif")
        png_file_path = os.path.join(png_folder, base_name + ".png")
        
        with rasterio.open(reference_geotiff_path) as src:
            profile = src.profile
            transform = src.transform
          
        profile.update(dtype=rasterio.uint8, count=1, compress='lzw')
    
        with rasterio.open(png_file_path) as src:
            png_data = src.read(1) 
    
        # Create a new GeoTIFF with the same georeference
        with rasterio.open(save_geotiff_path, 'w', **profile) as dst:
            dst.write(png_data, 1)
            
    return None
            
def masktif2vector(masktif_directory, output_gpkg):
    '''
    extract polygon vectors from masks, concatenate all, add img name & region fileds, and save as gpkg or shp. 

    Parameters
    ----------
    masktif_directory : TYPE
        folder containing all masks in tif.
    output_gpkg : TYPE
        output name. 

    output_shp : TYPE
        output name.

    Returns
    -------
    None.

    '''
    all_polygons_gdf = gpd.GeoDataFrame()

    # List all GeoTIFF files in the input directory
    mask_files = [f for f in os.listdir(masktif_directory) if f.endswith('.tif')]

    # Loop through each mask file
    for mask_file in tqdm(mask_files):
        mask_path = os.path.join(masktif_directory, mask_file)

        # Open the GeoTIFF file with Rasterio
        with rasterio.open(mask_path) as src:
            # Read the mask as a numpy array
            mask = src.read(1)
            
            #plt.imshow(mask, 'gray', interpolation='none')

            # Generate polygons from the binary mask
            shapes = features.shapes(mask, transform=src.transform, mask=(mask >= 1))

            # Create a GeoDataFrame from the polygons
            gdf = gpd.GeoDataFrame({'geometry': [shape(s) for s, v in shapes]})
            
            # add #img & # region
            # Create a DataFrame with new attribute data
             
            imname = mask_path.rsplit('/')[-1].rsplit('.')[0]
            assert imname.rsplit('_')[0] in ['0603','0619','0731','0825'], "ERROR: No image name found, check the image name."
            data = {'image': [replace_imname(imname.rsplit('_')[0])], 'region_num': [imname.rsplit('r')[-1]]}
            new_attributes_df = pd.DataFrame(data)
            
        
            # Create an empty list to store concatenated DataFrames
            new_attributes_df2 = []
            
            for _ in range(len(gdf)):
                # Append a copy of the original DataFrame to the list
                new_attributes_df2.append(new_attributes_df.copy())
            
            
            new_attributes_df2_ = pd.concat(new_attributes_df2, ignore_index=True)
            merged_gdf = gdf.merge(new_attributes_df2_, left_index=True, right_index=True)

                        
            # Append the polygons from this mask to a list
            all_polygons_gdf_list = [all_polygons_gdf, merged_gdf]
            all_polygons_gdf = pd.concat(all_polygons_gdf_list, ignore_index=True)

    #all_polygons_gdf.to_file(output_shp)
    all_polygons_gdf.to_file(output_gpkg, driver='GPKG', crs=src.crs)
        
    return None



# Specify the paths to the folders containing TIFF and PNG files
tiff_folder = args.img_dir
png_folder = args.pred_dir

# add georeference to png masks and save as tif in png_folder
#png2tif_w_ref_allin1folder(png_folder, tiff_folder) # predicted & post-processed

# convert masks to shp with fileds of #img and # region
masktif2vector(png_folder, args.save_path)

# filter out small polygons
gpd_train = gpd.read_file(args.train_path)
gpd_test = gpd.read_file(args.save_path)

gpd_train_area = gpd_train['geometry'].area
gpd_test_area = gpd_test['geometry'].area
min_area = gpd_train_area.min()

gpd_test = gpd_test[gpd_test_area > min_area]

gpd_test.to_file(args.save_path, driver='GPKG', crs=gpd_test.crs)

print("Final result saved to: ", args.save_path)

