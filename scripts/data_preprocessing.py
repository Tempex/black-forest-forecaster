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
    labels = labels.astype("object")
    if to_numbers:
        labels_int = np.zeros_like(labels)
        # Loop through each label
        for i in range(len(labels[1])):
            # Assign a 1 if it is a Rotbuche
            if labels[1][i] == 'Rotbuche':
                labels_int[:,i] =  1
            # Assign a 2 if it is Ahorn    
            elif labels[1][i] == 'Berg-Ahorn' or labels[1][i] == 'Feld-Ahorn':
                labels_int[:,i] = 2
            # Assign a 3 if it is a Birke    
            elif labels[1][i] == 'Haenge-Birke' or labels[1][i] == 'Moor-Birke':
                labels_int[:,i] = 3
        # The array is still of type "obj", we need to change it to "int"
        labels = labels_int.astype('int') 
    else:
        labels_sorted = np.full_like(labels,"Other")
        # Loop through each label
        for i in range(len(labels[1])):
            # Assign a 1 if it is a Rotbuche
            if labels[1][i] == 'Rotbuche':
                labels_sorted[:,i] = 'Rotbuche' 
            # Assign a 2 if it is Ahorn    
            elif labels[1][i] == 'Berg-Ahorn' or labels[1][i] == 'Feld-Ahorn':
                labels_sorted[:,i] = 'Ahorn'
            # Assign a 3 if it is a Birke    
            elif labels[1][i] == 'Haenge-Birke' or labels[1][i] == 'Moor-Birke':
                labels_sorted[:,i] = 'Birke'
        labels = labels_sorted     
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

def augment_images(img_array:np.ndarray)->np.ndarray:
    """Function to perform data augmentation on given array.

    Args:
        img_array (np.ndarray): array to be augmented

    Returns:
        np.ndarray: array with augmented data
    """
    # Predefine an array that will contain the augmented images
    img_array_full = np.tile(img_array, (12,1,1,1,1))

    # Loop through each image
    for i in range(img_array.shape[0]):

        # Augmentation number 0: Nothing
        img_aug = img_array_full[0,i,:,:,:]
        img_array_full[0,i,:,:,:] = img_aug

        # Augmentation number 1: 90° rotation        
        img_aug = tf.image.rot90(img_array[i], k=1).numpy()
        img_array_full[1,i,:,:,:] = img_aug

        # Augmentation number 2: 90° rotation + vertical flip
        img_aug = tf.image.rot90(img_array[i], k=1).numpy()
        img_aug = tf.image.flip_up_down(img_aug).numpy()
        img_array_full[2,i,:,:,:] = img_aug

        # Augmentation number 3: 90° rotation + horizontal flip
        img_aug = tf.image.rot90(img_array[i], k=1).numpy()
        img_aug = tf.image.flip_left_right(img_aug).numpy()
        img_array_full[3,i,:,:,:] = img_aug

        # Augmentation number 4: 90° rotation + vertical flip +
        # horizontal flip
        img_aug = tf.image.rot90(img_array[i], k=1).numpy()
        img_aug = tf.image.flip_up_down(img_aug).numpy()
        img_aug = tf.image.flip_left_right(img_aug).numpy()
        img_array_full[4,i,:,:,:] = img_aug

        # Augmentation number 5: 270° rotation
        img_aug = tf.image.rot90(img_array[i], k=3).numpy()
        img_array_full[5,i,:,:,:] = img_aug

        # Augmentation number 6: 270° rotation + vertical flip
        img_aug = tf.image.rot90(img_array[i], k=3).numpy()
        img_aug = tf.image.flip_up_down(img_aug).numpy()
        img_array_full[6,i,:,:,:] = img_aug

        # Augmentation number 7: 90° rotation + horizontal flip
        img_aug = tf.image.rot90(img_array[i], k=3).numpy()
        img_aug = tf.image.flip_left_right(img_aug).numpy()
        img_array_full[7,i,:,:,:] = img_aug

        # Augmentation number 8: 270° rotation + vertical flip +
        # horizontal flip
        img_aug = tf.image.rot90(img_array[i], k=3).numpy()
        img_aug = tf.image.flip_up_down(img_aug).numpy()
        img_aug = tf.image.flip_left_right(img_aug).numpy()
        img_array_full[8,i,:,:,:] = img_aug

        # Augmentation number 9: vertical flip + horizontal flip
        img_aug = tf.image.flip_up_down(img_array[i]).numpy()
        img_aug = tf.image.flip_left_right(img_aug).numpy()
        img_array_full[9,i,:,:,:] = img_aug

        # Augmentation number 10: vertical flip
        img_aug = tf.image.flip_up_down(img_array[i]).numpy()
        img_array_full[10,i,:,:,:] = img_aug

        # Augmentation number 11: horizontal flip
        img_aug = tf.image.flip_left_right(img_array[i]).numpy()
        img_array_full[11,i,:,:,:] = img_aug

        # Report progress
        if (i%1000 == 0) & (i != 0):
            print(' - ' + str(round(100*i/img_array.shape[0])) + 
            '% of all images have been augmented.')
        elif (i%100 == 0) & (i != 0):
            print('.', end='')
    return img_array_full

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
