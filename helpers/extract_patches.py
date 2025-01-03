from collections import defaultdict

import cv2
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import os
import glob
import sklearn.preprocessing
from tqdm import tqdm
import bz2
import pickle
from sklearn.decomposition import PCA

from logging_script import setup_logging

script_name = os.path.basename(__file__)
print(f"Running script: {script_name}")
logger = setup_logging(script_name)
import argparse
import pickle

OPENBLAS_NUM_THREADS = 1

IMG_EXTENSIONS = (
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.tiff', '.TIF', '.TIFF'
)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def black_pixels(img):
    return np.sum(img == 0)


def white_pixels(img):
    return np.sum(img == 255)


def gather_image_files_with_oswalk(directory, extensions):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                yield os.path.join(root, file)


# it returns the list of [(x,y), descriptors)] of the image for each key_point
# it takes the input image path and reads it using cv2.imread
def calc_features(input_img, arguments):
    img = cv2.imread(input_img)
    if arguments.scale != -1:
        scale = arguments.scale
        height = img.shape[0] * scale
        width = img.shape[1] * scale
        new_size_mask = (int(width), int(height))
        img = cv2.resize(img, new_size_mask, interpolation=cv2.INTER_CUBIC)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img_gray.copy()

    # create a SIFT object
    sift = cv2.SIFT_create(sigma=args.sigma)
    # detect key_points , kp is a list of key_points objects. each key_point object has the following attributes:
    #kp[0].pt -> (x,y) coordinates of the first key_point, kp[0].size -> size of the first key, kp[0].angle -> angle of the first key_point
    kp = sift.detect(img_gray, None)

    #performing image binarization using Otsu's thresholding method. Otsu's method automatically determines the optimal
    # threshold value to separate the foreground (e.g., objects) from the background.
    #The function returns two values:
    #_: The computed threshold value (ignored here by using _ since it's not needed).
    #img_bin: The binary image resulting from the thresholding operation

    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # filter key_points based on their size but not doesn't filter any key_point in this implementation
    # it takes the point object as input and returns the point object
    def filter_keypoint(point):
        # if 3 < point.size < 10:
        #     return point
        return point

    new_kp = []
    uniq_kp = []

    num_pixel = arguments.win_size ** 2
    for p in kp:
        # p.pt is a tuple of (x,y) coordinates of the key_point
        point = (int(p.pt[0]), int(p.pt[1]))
        # get_image_patch returns the patch of the image centered at the key_point
        roi = get_image_patch(img, point[0], point[1], arguments)

        if roi is None:
            continue

        if arguments.centered:
            if img_bin[point[1], point[0]] != 0:
                continue

        if arguments.black_pixel_thresh != -1:
            if black_pixels(roi) > (arguments.black_pixel_thresh * num_pixel):
                continue

        if arguments.white_pixel_thresh != -1:
            if white_pixels(roi) > (arguments.white_pixel_thresh * num_pixel):
                continue

        if not point in uniq_kp and filter_keypoint(p):
            # add the key_point object to the list of key_points
            new_kp.append(p)
            # add the key_point coordinates (x,y) to the list of uniq_kp
            uniq_kp.append(point)

    if not new_kp:
        return []
    # compute the descriptors of the key_points
    # img_gray is the image of shape (height, width) and new_kp is the list of key_points
    # it returns a tuple of (key_points, descriptors)
    _, desc = sift.compute(img_gray, new_kp)
    # TODO STEP 1.1: descriptors are normalized with the Hellinger kernel (elementwise square root followed by l1-normalization)
    desc = sklearn.preprocessing.normalize(desc, norm='l1')
    desc = np.sign(desc) * np.sqrt(np.abs(desc))
    desc = sklearn.preprocessing.normalize(desc, norm='l2')

    # p is a tuple of (x,y) coordinates , desc[i] is the descriptor of the keypoint p and it is a 128-d vector
    # it is a list of tuples of (key_point, descriptor) for each key_point in the image
    kp_tupple = [(p, desc[i]) for i, p in enumerate(uniq_kp)]

    return kp_tupple


def get_image_patch(img, px, py, arguments):
    half_win_size = int(arguments.win_size / 2)
    if not (half_win_size < px < img.shape[1] - half_win_size and half_win_size < py < img.shape[0] - half_win_size):
        # if px - half_win_size < 0 or px + half_win_size > img.shape[1] or py - half_win_size < 0 or py + half_win_size > img.shape[0]:
        return None

    roi = img[py - half_win_size:py + half_win_size, px - half_win_size:px + half_win_size]
    assert roi.shape == (half_win_size * 2, half_win_size * 2), 'shape of the roi is not (%d,%d). It is (%d,%d)' % \
                                                                (half_win_size * 2, half_win_size * 2,
                                                                 roi.shape[0], roi.shape[1])
    return roi


def extract_patches(filename, tup, args):
    # tup is a list of tuples [((x,y), cluster_label),....]
    if len(tup) > args.patches_per_page and args.patches_per_page != -1:
        idx = np.linspace(0, len(tup) - 1, args.patches_per_page, dtype=np.int32)
        tup = [tup[i] for i in idx]
    # unpack the tup
    # points is a tuple of ((x1,y1),(x2,y2),....) and clusters is a tuple of (cluster_label1, cluster_label2,....)
    points, clusters = zip(*tup)

    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if args.scale != -1:
        scale = args.scale
        height = img.shape[0] * scale
        width = img.shape[1] * scale
        new_size_mask = (int(width), int(height))
        img = cv2.resize(img, new_size_mask, interpolation=cv2.INTER_CUBIC)

    # Dictionary to store patches by cluster
    clustered_patches = defaultdict(list)

    count = 0
    for p, c in zip(points, clusters):
        roi = get_image_patch(img, p[0], p[1], args)
        if roi is None:
            logger.info(f"no roi found #iteration {count} for {filename}")
            continue

        file_name = str(c) + '_' + os.path.splitext(os.path.basename(filename))[0] + '_' + str(count)
        int_key_C = int(c)
        # Append patch and its metadata to the corresponding cluster
        try:
            clustered_patches[int_key_C].append({
                "patch": roi,  # The patch as a NumPy array
                "keypoint": p,  # Keypoint coordinates
                "patch_name": file_name  # Original image filename
            })
            count += 1
        except Exception as e:
            # print("exception accused during writing patches into the dict")
            logger.info("exception while adding patches on dictionary inside the extract_patches function")

    # logger.info(f'Extracted {sum(len(patches) for patches in clustered_patches.values())} patches from {filename}')
    # print(f"{count} number of patches added for {filename}")
    return clustered_patches


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="extract patches from images")
    parser.add_argument('--in_dir', metavar='in_dir', dest='in_dir', type=str, nargs=1,
                        help='input directory', required=True)
    parser.add_argument('--out_dir', metavar='out_dir', dest='out_dir', type=str, nargs=1,
                        help='output directory', required=True)
    parser.add_argument('--win_size', metavar='win_size', dest='win_size', type=int, nargs='?',
                        help='size of the patch',
                        default=32)
    parser.add_argument('--num_of_clusters', metavar='num_of_clusters', dest='number_of_clusters', type=int, nargs='?',
                        help='number of clusters',
                        default=-1)
    parser.add_argument('--patches_per_page', metavar='patches_per_page', dest='patches_per_page', type=int, nargs='?',
                        help='maximal number of patches per page (-1 for no limit)',
                        default=-1)
    parser.add_argument('--scale', type=float, help='scale images up or down',
                        default=-1)
    parser.add_argument('--sigma', type=float, help='blur factor for SIFT',
                        default=1.6)
    parser.add_argument('--black_pixel_thresh', type=float,
                        help='if more black_pixel_thresh percent of the pixels are black -> discard',
                        default=0.5)
    parser.add_argument('--white_pixel_thresh', type=float,
                        help='if more than white_pixel_thresh percent of the pixels are white -> discard',
                        default=0.5)
    parser.add_argument('--centered', type=bool,
                        help='filter patches whose keypoints are not located on handwriting, only for binarized datasets',
                        default=True)

    args = parser.parse_args()

    assert os.path.exists(args.in_dir[0]), 'in_dir {} does not exist'.format(args.in_dir[0])

    # creating the output directory if it doesn't exist
    if not os.path.exists(args.out_dir[0]):
        logger.info('creating directory %s' % args.out_dir[0])
        os.mkdir(args.out_dir[0])

    assert len(os.listdir(args.out_dir[0])) == 0, 'out_dir is not empty'
    assert args.win_size % 2 == 0, 'win_size must be even'
    num_cores = int(multiprocessing.cpu_count() / 2)
    # num_cores = 14
    path_to_centers = ''

    # Gather all image files
    files = list(gather_image_files_with_oswalk(args.in_dir[0], IMG_EXTENSIONS))

    assert len(files) > 0, 'no images found'
    logger.info('Found {} images'.format(len(files)))

    logger.info('calculating features for images in %s (number of cores:%d)' % (args.in_dir, num_cores))
    results = []
    # it sends the file path to calculate the features of the image
    # TODO Step 1: detect keypoint and corresponding descriptors.
    results = Parallel(n_jobs=num_cores, verbose=9)(delayed(calc_features)(f, args) for f in files)

    logger.info('collecting descriptors')
    # it contains the descriptors of the key_points
    desc_list = []
    # it contains the (x,y) coordinates of the key_points
    kp_list = []
    # it contains the files names
    fn_list = []
    for r, f in tqdm(zip(results, files)):
        if len(r) == 0:
            logger.warning('no keypoints found in file {} '.format(f))
        for kp, desc in r:
            kp_list.append(kp)
            desc_list.append(desc)
            fn_list.append(f)
    results = None

    logger.info('calculating pca (number of patches: {})'.format(len(desc_list)))

    # TODO STEP 2: dimensionality reduction via PCA from 128 to 32
    pca = PCA(32, whiten=True)
    desc = pca.fit_transform(np.array(desc_list))
    desc_list = None

    logger.info('calculating new centers (shape: {})'.format(desc.shape))

    logger.info("starting KMeans (centers: {})".format(args.number_of_clusters))
    # km = KMeans(n_clusters=number_of_clusters, random_state=0, n_jobs=-1, verbose=0).fit(desc)
    import sklearn.cluster

    # creating the kmeans object
    # TODO STEP 3 #: cluster the descriptors via k-means in 5000 clusters
    kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=args.number_of_clusters, compute_labels=False,
                                             batch_size=10000 if args.number_of_clusters <= 1000 else 50000)

    if desc.shape[0] > 500000:
        idx = np.linspace(0, desc.shape[0] - 1, 500000, dtype=np.int32)
        kmeans.fit(desc[idx])
    else:
        kmeans.fit(desc)

    # save the centers to the output directory as a pickle file
    center_path = os.path.join(args.out_dir[0], 'centers.pkl')
    logger.info('saving centers to %s' % center_path)
    pickle.dump(kmeans, open(center_path, 'wb'))

    # for each image file it contains { filename:[((x,y), cluster_label) , ((x,y), cluster_label), ...]}
    patches_files = {}
    feature_count = 0
    batch_size = 50000
    for batch_start in tqdm(range(0, desc.shape[0], batch_size), 'Transforming and filtering'):
        def batch(d):
            return d[batch_start:batch_start + batch_size]


        dist = kmeans.transform(batch(desc))
        prediction = kmeans.predict(batch(desc))
        #TODO STEP 3.1 # we filter keypoints whose descriptors 'd' violate ||d-m1||/||d-m2|| >0.9
        # it filter key points that lay near the border of two different clusters - those
        # are therefore considered to be ambiguous
        dist = np.sort(dist)
        ratio = dist[:, 0] / dist[:, 1]

        for p, f, c, r in zip(batch(kp_list), batch(fn_list), prediction, ratio):
            if r <= 0.9:
                feature_count += 1
                if f in patches_files:
                    patches_files[f].append((p, c))
                else:
                    patches_files[f] = [(p, c)]

    logger.info('copying %i (all patches per page) image patches to %s ' % (feature_count, args.out_dir[0]))
    #tup is list of tuples of [((x,y), cluster_label),....]
    #this function extracts the patches from the image and saves them to the output directory

    # TODO STEP 4 #The 32 × 32 patch is extracted at the keypoint location
    patches_results = Parallel(n_jobs=num_cores, verbose=9)(
        delayed(extract_patches)(filename, tup, args) for filename, tup in patches_files.items())

    # print(f"type of results {type(patches_results)}")
    # print(f"len(results) = {len(patches_results)}")
    # Global dictionary to hold consolidated patches by cluster

    all_clusters = defaultdict(list)
    # Consolidate patches from all images into the global cluster dictionary
    for image_patches in patches_results:
        for cluster_id, patches in image_patches.items():
            all_clusters[cluster_id].extend(patches)
        # print(f"{len(image_patches)} added to cluster {cluster_id}")# Append patches to the respective cluster

    # Save the consolidated patches to a single file
    output_file = os.path.join(args.out_dir[0], "consolidated_clusters.pkl.bz2")
    with bz2.BZ2File(output_file, "wb") as f:
        pickle.dump(all_clusters, f)

    config_out_path = os.path.join(args.out_dir[0], 'db-creation-parameters.json')
    logger.info(f'writing config parameters to {config_out_path}')
    with open(config_out_path, 'w') as f:
        import json

        json.dump(vars(args), f)
    logger.info('done')
