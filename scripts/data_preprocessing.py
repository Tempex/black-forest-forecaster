import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def prepare_labels(labels:np.ndarray,to_numbers:bool=True)->np.ndarray:
    """Reclassifies the trees into 4 classes and optionally encodes them numerically.
    Classes are: "0: Other", "1: Ahorn", "2: Birke", "3: Rotbuche"

    Args:
        labels (np.ndarray): Array of tree labels
        to_numbers (bool, optional): If True encodes the labels numerically. Defaults to True.

    Returns:
        np.ndarray: Array with the reclassified and encoded labels.
    """
    orig_labels = ['Sal-Weide', 'Vogelbeere', 'Haenge-Birke', 'Asch-Weide',
       'Grau-Erle', 'Berg-Ahorn', 'Feld-Ahorn', 'Gem. Hasel', 'Rotbuche', 
       'Vogelkirsche', 'Traubeneiche', 'Schwarzer Holunder',
       'Roteiche', 'Rosskastanie', 'Berg-Ulme', 'Schwarz-Erle',
       'Moor-Birke', 'Espe', 'Stechpalme', 'Schwarz-Weide',
       'Gew. Traubenkirsche', 'Linde', 'Weide', 'Erle']
    new_labels = ['Other', 'Other', 'Birke', 'Other',
        'Other', 'Ahorn', 'Ahorn', 'Other', 'Rotbuche',
        'Other', 'Other', 'Other', 
        'Other', 'Other', 'Other', 'Other', 
        'Birke', 'Other', 'Other', 'Other', 
        'Other', 'Other', 'Other', 'Other']
    for i in range(labels.shape[0]):
        for old, new in zip(orig_labels,new_labels):
            labels[i] = np.char.replace(labels[i],old,new)
        if to_numbers:
            label_dict = {"Other":"0","Ahorn":"1","Birke":"2","Rotbuche":"3"}
            for k, v in label_dict.items():
                labels[i] = np.char.replace(labels[i],k,v)
    labels = labels.astype(int)         
    return labels

def tf_train_test_val_split(data:np.ndarray, labels:np.ndarray, train_size:float=0.8, val_size:float=0.1, random_state:int=None)->np.ndarray:
    """Splits the Data into train, test and validation sets, stratified by labels.

    Args:
        data (np.ndarray): The data to be split
        labels (np.ndarray): Labels for the data to be split
        train_size (float, optional): Training set size. Defaults to 0.8.
        val_size (float, optional): Size of the validation set. Defaults to 0.1.
        random_state (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        np.ndarray: Training data
        np.ndarray: Test data
        np.ndarray: Validation data
        np.ndarray: Training labels
        np.ndarray: Test labels
        np.ndarray: Validation labels
    """
    nr_channels = data.shape[-1]

    # Reshape data to combine the first two dimensions
    reshaped_data = data.reshape(-1, 35, 35, nr_channels)
    reshaped_labels = labels.reshape(-1)

    # Generate indices for splitting
    idx_train, idx_temp, labels_train, labels_temp = train_test_split(np.arange(len(reshaped_data)), reshaped_labels, train_size=train_size, stratify=reshaped_labels, random_state=random_state)
    idx_val, idx_test, labels_val, labels_test = train_test_split(idx_temp, labels_temp, test_size=val_size / (1 - train_size), stratify=labels_temp, random_state=random_state)

    # Split the data and reshape back
    data_train = reshaped_data[idx_train]
    data_val = reshaped_data[idx_val]
    data_test = reshaped_data[idx_test]

    # Reshape labels arrays
    labels_train = labels_train.reshape(-1)
    labels_val = labels_val.reshape(-1)
    labels_test = labels_test.reshape(-1)
    return data_train, data_test, data_val, labels_train, labels_test, labels_val  

def oversample(train:np.ndarray, label_train:np.ndarray)->np.ndarray:
    """Oversamples the give dataset by random minority oversampling. All classes will be oversampled till they have the same number as the majority class.

    Args:
        train (np.ndarray): Dataset to be oversampled
        label_train (np.ndarray): Labels to be oversampled

    Returns:
        np.ndarray: Oversampled data
        np.ndarray: Oversampled labels

    """
    class_counts = np.bincount(label_train) # Count occurrences of each class
    n_max = np.max(class_counts)  # Get the count of the majority class

    img_os = train.copy()
    label_os = label_train.copy()

    for i, n_class in enumerate(class_counts):
        if n_class != n_max:
            class_indices = np.where(label_train == i)[0]

            # Calculate the number of additional samples needed
            n_samples_needed = n_max - n_class

            # Randomly sample indices with replacement
            rand_indices = np.random.choice(class_indices, size=n_samples_needed, replace=True)

            # Append the randomly selected samples to the oversampled arrays
            img_os = np.append(img_os, train[rand_indices], axis=0)
            label_os = np.append(label_os, label_train[rand_indices], axis=0)
    return img_os, label_os

def create_datasets(train:np.ndarray, test:np.ndarray, val:np.ndarray, l_train:np.ndarray, l_test:np.ndarray, l_val:np.ndarray)->tf.data.Dataset:
    """Creates Tensorflow datasets from the given arrays.

    Args:
        train (np.ndarray): Training data
        test (np.ndarray): Test data
        val (np.ndarray): Validation data
        l_train (np.ndarray): Training labels
        l_test (np.ndarray): Test labels
        l_val (np.ndarray): Validation labels

    Returns:
        tf.data.Dataset: Training dataset containing data and labels
        tf.data.Dataset: Test dataset containing data and labels
        tf.data.Dataset: Validation dataset containing data and labels
    """
    train_ds = tf.data.Dataset.from_tensor_slices((train,l_train))
    test_ds = tf.data.Dataset.from_tensor_slices((test,l_test))
    val_ds = tf.data.Dataset.from_tensor_slices((val,l_val))
    return train_ds, test_ds, val_ds

if __name__=="__main__":
    pass
