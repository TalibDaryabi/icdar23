import bz2
import io
import pickle

import torch.utils.data as data

from PIL import Image
import os
import os.path
from tqdm import tqdm
import numpy as np
import re
from PIL import ImageOps
import glob
from torch.utils.data import DataLoader
import logging
import os
import pickle
import bz2
import re
from tqdm import tqdm


def make_dataset_from_pickle_file(folder_path):
    """
    Extracts filenames, patches, and keypoints from the first .pkl.bz2 file found in a folder.

    Args:
        folder_path (str): Path to the folder containing the .pkl.bz2 file.

    Returns:
        tuple: Three lists containing filenames, patches, and keypoints, respectively.
    """
    # Ensure the folder exists
    if not os.path.isdir(folder_path):
        raise ValueError(f"The folder {folder_path} does not exist.")

    # Find the first .pkl.bz2 file in the folder
    pickle_file = next((f for f in os.listdir(folder_path) if f.endswith(".pkl.bz2")), None)
    if pickle_file is None:
        raise ValueError(f"No .pkl.bz2 file found in the folder {folder_path}.")

    file_path = os.path.join(folder_path, pickle_file)

    # Load the pickle file
    try:
        with bz2.BZ2File(file_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load the pickle file: {file_path}. Error: {e}")

    # Lists to store filenames, patches, and keypoints
    file_names = []
    patches_of_files = []
    key_points_of_patches = []

    # Extract data
    for cluster_id, patches in data.items():
        for patch_data in patches:
            file_names.append(patch_data["patch_name"])
            patches_of_files.append(patch_data["patch"])
            key_points_of_patches.append(patch_data["keypoint"])

    # Sort filenames and reorder other lists accordingly
    sorted_indices = sorted(range(len(file_names)), key=lambda i: file_names[i])
    file_names = [file_names[i] for i in sorted_indices]
    patches_of_files = [patches_of_files[i] for i in sorted_indices]
    key_points_of_patches = [key_points_of_patches[i] for i in sorted_indices]

    return file_names, patches_of_files, key_points_of_patches


def extract_labels_from_patches(files, rxs):
    """
    Extracts labels, label-to-integer mappings, and integer-to-label mappings from patch names.

    Args:
        files (list): List of patch names (from make_dataset_from_pickle_file).
        rxs (dict): Dictionary of label names and corresponding regex patterns.

    Returns:
        tuple: (files, labels, label_to_int, int_to_label)
            - files (list): List of patch names.
            - labels (dict): Dictionary of labels, converted to integer mappings for each regex.
            - label_to_int (dict): Mapping from labels to integers for each regex.
            - int_to_label (dict): Mapping from integers to labels for each regex.
    """
    # Initialize dictionaries
    labels = {}
    label_to_int = {}
    int_to_label = {}

    # Process each patch name
    for patch_name in tqdm(files, desc="Extracting Labels"):
        for name, regex in rxs.items():
            # Match the regex
            match = re.search(regex, patch_name)
            if not match:
                continue  # Skip if regex does not match

            # Extract the label
            extracted_label = '_'.join(match.groups())

            # Initialize dictionaries for the current label name
            labels[name] = labels.get(name, [])
            label_to_int[name] = label_to_int.get(name, {})
            int_to_label[name] = int_to_label.get(name, {})

            # Map label to integer
            if extracted_label not in label_to_int[name]:
                new_int = len(label_to_int[name])
                label_to_int[name][extracted_label] = new_int
                int_to_label[name][new_int] = extracted_label

            # Append the integer label
            labels[name].append(label_to_int[name][extracted_label])

    return labels, label_to_int, int_to_label


def make_dataset(cur_dir, rxs):
    """
    # files = ["789_123-IMG_MAX_456_7", "890_124-IMG_MAX_457_8"]
     {                                                     label_to_int      {                          int_to_label  {
      'cluster': [0, 1],  # Mapped integers for clusters            'cluster': {'789': 0, '890': 1},                    'cluster': {0: '789', 1: '890'},
      'writer': [0, 1],   # Mapped integers for writers              'writer': {'123': 0, '124': 1},                     'writer': {0: '123', 1: '124'},
      'page': [0, 1]      # Mapped integers for pages                'page': {'456': 0, '457': 1}                       'page': {0: '456', 1: '457'}
       }                                                             }                                                   }
    """
    assert rxs is not None, 'no regular expression is set'
    files_names, file_patches, file_key_points = make_dataset_from_pickle_file(cur_dir)

    if len(files_names) == 0:
        raise (RuntimeError("Found 0 images data in pickle file"))

    labels, label_to_int, int_to_label = extract_labels_from_patches(files_names, rxs)
    return files_names, file_patches, file_key_points, labels, label_to_int, int_to_label


def pil_loader_from_patch(patch):
    # If the patch is a NumPy array, convert it to a PIL image
    if isinstance(patch, np.ndarray):
        img = Image.fromarray(patch)

    # If the patch is raw bytes, load it using BytesIO
    elif isinstance(patch, (bytes, bytearray)):
        img = Image.open(io.BytesIO(patch))
    else:
        raise ValueError("Unsupported patch format. Must be NumPy array or bytes.")

    # Standardize the image
    if len(img.mode) > 1:  # If the image has multiple channels
        return ImageOps.grayscale(img.convert('RGB'))  # Convert to RGB and then grayscale

    return img.convert(mode='L')  # Convert to grayscale directly if single-channel


class WrapableDataset(data.Dataset):
    # Defines a common interface for working with datasets. It inherits from torch.utils.data.Dataset
    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def supported_classes():
        return {'CombineLabels': 'CombineLabels',
                'SelectLabels': 'SelectLabels',
                'TransformImages': 'TransformImages',
                'Sample': 'Sample',
                'ClassSampler': 'ClassSampler'
                }

    def _get_wrapper_class_constructor(self, name):
        def wrapper(*args, **kw):
            c = self.supported_classes()[name]
            if type(c) == str:
                return globals()[c](self, *args, **kw)
            else:
                return c(self, *args, **kw)

        return wrapper

    def __getattr__(self, attr):
        if attr in self.supported_classes():
            return self._get_wrapper_class_constructor(attr)

    def __getitem__(self, index):
        #TODO outermost wrapper
        return self.get_image(index), self.get_label(index)


class DatasetWrapper(WrapableDataset):
    # Base Wrapper
    def __getattr__(self, attr):
        if attr in self.supported_classes():
            return self._get_wrapper_class_constructor(attr)
        else:
            return getattr(self.dataset, attr)

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)


class Sample(DatasetWrapper):
    # sample a subset of the dataset
    def __init__(self, *args, samples=None, **kwargs):
        super().__init__(*args, **kwargs)

        if not samples:
            samples = len(self.dataset)

        self.samples = min(samples, len(self.dataset))  # we don't want to sample more than we have
        self.idx = np.linspace(0, len(self.dataset) - 1, self.samples, dtype=np.int32)

    def get_label(self, index):
        return self.dataset.get_label(self.idx[index])

    def get_image(self, index):
        return self.dataset.get_image(self.idx[index])

    def __len__(self):
        return len(self.idx)


class CombineLabels(DatasetWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.packed_labels = np.array([np.asscalar(np.argwhere(np.all(self.unique_labels == l, axis=1))) for l in
                                       self.packed_labels]).astype('int32')
        self.unique_labels = np.unique(self.packed_labels, axis=0)

    def get_label(self, index):
        return self.packed_labels[index]


class SelectLabels(DatasetWrapper):
    # Allows you to select a subset of labels from the dataset
    def __init__(self, *args, label_names, **kwargs):
        super().__init__(*args, **kwargs)

        self.label_names = label_names if type(label_names) == list else [label_names]

        self.packed_labels = np.stack([self.labels[l] for l in self.label_names], axis=1)
        self.unique_labels = np.unique(self.packed_labels, axis=0)

    def get_label(self, index):
        label = tuple(self.packed_labels[index])

        if len(label) == 1:
            label = label[0]

        return label

    def __getitem__(self, index):
        return self.dataset.get_image(index), self.get_label(index)


class ClassSampler(DatasetWrapper):
    def __init__(self, *args, label_names, **kwargs):
        super().__init__(*args, **kwargs)

        self.label_names = label_names if type(label_names) == list else [label_names]
        # labels_ids = label_ids if type(label_ids) == list else [label_ids]

        self.packed_labels = np.stack([self.labels[l] for l in self.label_names], axis=1)
        self.unique_labels = np.unique(self.packed_labels, axis=0)

        self.indices_per_label = []
        print('Extracting subsets for Class Sampler')
        for ix in range(self.unique_labels.shape[0]):
            label_ids = self.unique_labels[ix]
            hits = np.sum(np.equal(self.packed_labels, label_ids), axis=1)
            self.indices_per_label.append(np.where(hits == len(label_ids))[0].tolist())

    def get_label(self, index):
        label = tuple(self.packed_labels[index])

        if len(label) == 1:
            label = label[0]

        return label

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, index):
        index = self.indices[index]
        return self.dataset.get_image(index), self.get_label(index)


class TransformImages(DatasetWrapper):
    def __init__(self, *args, transform, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform

    def get_image(self, index):
        img = self.dataset.get_image(index)
        img = img.convert('RGB')
        img = self.transform(img).float()
        return img


class ImageFolder(WrapableDataset):
    def __init__(self, dataset_name, dataset_type, path=None, regex=None, mean=None, loader='PIL',
                 data_augmentation=False):
        if path is None:
            logging.info("path parameter is none , inside the ImageFolder __init__")
            raise Exception
        logging.info('Loading {} {}dataset from {}'.format(dataset_name, dataset_type, path))

        files_names, file_patches, file_key_points, labels, label_to_int, int_to_label = make_dataset(path, regex)
        self.files_names = files_names
        self.file_patches = file_patches
        self.file_key_points = file_key_points
        self.label_to_int = label_to_int
        self.int_to_label = int_to_label
        self.label_names = [name for name, _ in regex.items()]
        self.packed_labels = np.stack([labels[l] for l in self.label_names], axis=1)

        self.labels = labels
        self.root = path
        self.regex = regex
        self._mean = mean

    @property
    def mean(self):
        if type(self._mean) == str:
            self._mean = np.load(os.path.join(self.root, self._mean))
        elif self._mean is None:
            cur_data = DataLoader(self, batch_size=min(1000, len(self)), shuffle=False, num_workers=8)

            mean = None
            logging.info('Calculating mean image for "{}"'.format(self.root))

            cnt = 0
            for img, _ in tqdm(cur_data, 'Calculating Mean'):
                s = img.size(0)
                m = np.mean(img.numpy(), axis=0)

                if mean is None:
                    mean = m
                else:
                    mean = mean + (m - mean) * s / (s + cnt)

                cnt += s
            self._mean = mean
        return self._mean

    def get_image(self, index):
        if not self.file_patches:
            raise ValueError("file_patches is empty or None.")
        # img_filename = self.files_names[index]
        img_patch = self.file_patches[index]
        assert img_patch is not None
        img = pil_loader_from_patch(img_patch)
        assert img is not None
        return img

    def get_label(self, index):
        label = tuple(self.packed_labels[index])
        if len(label) == 1:
            label = label[0]
        return label

    def __len__(self):
        return len(self.files_names)
