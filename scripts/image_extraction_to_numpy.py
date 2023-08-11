import pandas as pd
import numpy as np
import rasterio as rio
import glob
from typing import Union

def load_tree_coordinates(path:str, **tree_list:list)->tuple:
    """Function to load tree coordinates from a csv file and returns 
    a tuple with the tree coordinates and labels.

    Args:
        path (str): path to the .csv file containing the tree data
        tree_list (list, optional): List specifying which tree labels to extract.

    Returns:
        tuple: tuple containing tuples with (x_coordinate, y_coordinate, tree_label)
    """
    # Report that tree coordinates will be loaded
    print('Loading tree coordinates...')

    # Import data that contains the labeled but uncorrected gps tree
    # coordinates
    tc_df = pd.read_csv(path)
    # Extrac those variables that will be of importance
    tc_df = tc_df[['X', 'Y', 'desc']]
    # Rename the columns
    tc_df.columns = ['x_geo', 'y_geo', 'label']
    # Return the data frame
    if tree_list: # filter for trees specified in tree_list
        tc_df_filtered = tc_df[tc_df['label'].isin(tree_list)]
        data = tuple(tc_df_filtered.itertuples(index=False,name=None))
    else: 
        data = tuple(tc_df.itertuples(index=False,name=None))
    print("Done",end='\n')
    return data

def extract_trees(filepath:str,coordinates:tuple,rgb:bool=False)->np.ndarray:
    """Extracts pictures of trees from a .tif file and returns them as a numpy array.
    Picture size is 35 x 35 centered on the tree-coordinate.

    Args:
        filepath (str): Path to the .tif from where the trees should be extracted
        coordinates (tuple): Tuple containing the x & y coordinates and labels of the trees as tuples. 
        rgb (bool, optional): If True only extracts a RGB picture of the tree, else a RGB+IR picture. Defaults to False.

    Returns:
        img_array (np.ndarray): Array containing the images
        label_array (np.ndarray): Array containing the labels

    """
    print("Extracting trees...")
    save_list = []
    labels_list = []
    with rio.open(filepath) as data:
        for (lon, lat, label) in coordinates:
            # Get pixel coordinates from map coordinates
            py, px = data.index(lon, lat)
            window = rio.windows.Window(px - 14, py - 14, 35, 35)
            # Read the data in the window
            clip = data.read(window=window)
            clip = np.transpose(clip,(1,2,0))
            if rgb:
                clip = clip[:,:,:3]
                if clip.shape != (35,35,3):
                    continue
                else:
                    save_list.append(clip)
                    labels_list.append(label)
            else:
                if clip.shape != (35,35,4):
                    continue
                else:
                    save_list.append(clip)
                    labels_list.append(label)
            
    img_array = np.array(save_list)
    labels_array = np.array(labels_list)
    return img_array, labels_array

def save_data(savepath:str, filename:str, pic_array_list:Union[list,np.ndarray],label_array_list:Union[list,np.ndarray]):
    """Saves the extracted trees and labels in a folder as .npz file containing 2 arrays named "trees" and "labels".

    Args:
        savepath (str): Folder in which the file should be saved
        filename (str): Filename for the .npz file
        pic_array_list (Union[list,np.ndarray]): List or array of the images
        label_array_list (Union[list,np.ndarray]): List or array of the labels
    """
    final_pics = np.array(pic_array_list)
    final_labels = np.array(label_array_list)
    np.savez(f"{savepath}/{filename}",trees=final_pics,labels=final_labels)
    print(f"Array saved as '{filename}.npz' in '{savepath}'")

if __name__=="__main__":
    tree_data = load_tree_coordinates('./data/Laubbäume_utm32_cleaned.csv')
    filelist = glob.glob("./data/TDOP/*.tif")
    pic_arr_list = []
    label_arr_list = []
    for file in filelist:
        print(f"Processing:{file}...")
        pic_arr, label_arr = extract_trees(file,tree_data,rgb=True)
        pic_arr_list.append(pic_arr)
        label_arr_list.append(label_arr)
        print("Done", end='\n')

    save_data("./data/","data_rgb",pic_arr_list,label_arr_list)
