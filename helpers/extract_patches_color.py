import shutil
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
import argparse
import pickle

from utils.logging_script import setup_logging

script_name = os.path.basename(__file__)
print(f"Running script: {script_name}")
logger = setup_logging(script_name)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.tiff', '.TIF', '.TIFF'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def edge_pixels(mask):
    return np.sum(mask == 255)


def calc_features(input_img, arguments):
    img = cv2.imread(input_img)

    if arguments.scale != -1:
        scale = arguments.scale
        height = img.shape[0] * scale
        width = img.shape[1] * scale
        new_size_mask = (int(width), int(height))
        img = cv2.resize(img, new_size_mask, interpolation=cv2.INTER_CUBIC)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.Canny(img, 30, 200)

    sift = cv2.SIFT_create(sigma=args.sigma)
    kp = sift.detect(img_gray, None)

    def filter_keypoint(point):
        # if 3 < point.size < 10:
        #     return point
        return point

    new_kp = []
    uniq_kp = []

    num_pixel = arguments.win_size ** 2

    for p in kp:
        point = (int(p.pt[0]), int(p.pt[1]))

        roi = get_image_patch(img, point[0], point[1], arguments)
        mask_roi = get_image_patch(mask, point[0], point[1], arguments)

        if roi is None:
            continue

        if arguments.edge_pixels != -1:
            if edge_pixels(mask_roi) < (arguments.edge_pixels * num_pixel):
                continue

        if not point in uniq_kp and filter_keypoint(p):
            new_kp.append(p)
            uniq_kp.append(point)

    # print(f'{len(new_kp)} kps for {input_img}')

    _, desc = sift.compute(img_gray, new_kp)

    if not new_kp:
        desc = np.zeros(128)
        kp_tupple = [((int(img.shape[1] / 2), int(img.shape[0] / 2)), desc)]
        print(f'Nothing found for {input_img}')
        return kp_tupple

    desc = sklearn.preprocessing.normalize(desc, norm='l1')
    desc = np.sign(desc) * np.sqrt(np.abs(desc))
    desc = sklearn.preprocessing.normalize(desc, norm='l2')

    kp_tupple = [(p, desc[i]) for i, p in enumerate(uniq_kp)]

    if args.patches_per_page != -1 and (len(kp_tupple) > args.patches_per_page):
        idx = np.linspace(0, len(kp_tupple) - 1, args.patches_per_page, dtype=np.int32)
        kp_tupple = [kp_tupple[i] for i in idx]

    logger.info("Found %i keypoints in %s" % (len(kp_tupple), input_img))
    return kp_tupple


def get_image_patch(img, px, py, arguments):
    half_win_size = int(arguments.win_size / 2)
    if not (half_win_size < px < img.shape[1] - half_win_size and half_win_size < py < img.shape[0] - half_win_size):
        return None

    roi = img[py - half_win_size:py + half_win_size, px - half_win_size:px + half_win_size]
    assert roi.shape[:2] == (half_win_size * 2, half_win_size * 2), 'shape of the roi is not (%d,%d). It is (%d,%d)' % \
                                                                    (half_win_size * 2, half_win_size * 2,
                                                                     roi.shape[0], roi.shape[1])
    return roi


def extract_patches(filename, tup, args):
    if len(tup) > args.patches_per_page and args.patches_per_page != -1:
        idx = np.linspace(0, len(tup) - 1, args.patches_per_page, dtype=np.int32)
        tup = [tup[i] for i in idx]

    points, clusters = zip(*tup)

    img = cv2.imread(filename)

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
            print("continue")
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
            print("exception accused during writing patches into the dict")
            logger.info("exception while adding patches on dictionary inside the extract_patches function")

    logger.info(f'Extracted {sum(len(patches) for patches in clustered_patches.values())} patches from {filename}')
    print(f"{count} number of patches added for {filename}")
    return clustered_patches


def gather_image_files_with_oswalk(directory, extensions):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                yield os.path.join(root, file)


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
    parser.add_argument('--edge_pixels', type=float,
                        help='if more black_pixel_thresh percent of the pixels are black -> discard',
                        default=0.1)

    args = parser.parse_args()

    assert os.path.exists(args.in_dir[0]), 'in_dir {} does not exist'.format(args.in_dir[0])

    if not os.path.exists(args.out_dir[0]):
        logger.info('creating directory %s' % args.out_dir[0])
        os.mkdir(args.out_dir[0])

    assert len(os.listdir(args.out_dir[0])) == 0, 'out_dir is empty'

    assert args.win_size % 2 == 0, 'win_size must be even'

    num_cores = int(multiprocessing.cpu_count() / 2)
    num_cores = 10
    path_to_centers = ''

    # Gather all image files
    # files = list(gather_image_files_with_oswalk(args.in_dir[0], IMG_EXTENSIONS))
    files = [f for f in glob.glob(args.in_dir[0] + '/**/*.*', recursive=True) if os.path.isfile(f) and is_image_file(f)]

    assert len(files) > 0, 'no images found'
    logger.info('Found {} images'.format(len(files)))

    # Step 1: Extract features from images
    logger.info('calculating features for images in %s (number of cores:%d)' % (args.in_dir, num_cores))
    results = []
    results = Parallel(n_jobs=num_cores, verbose=9)(delayed(calc_features)(f, args) for f in files)

    logger.info('collecting descriptors')
    desc_list = []
    kp_list = []
    fn_list = []
    # Step 2: Collect descriptors and key_points
    for r, f in tqdm(zip(results, files)):
        if len(r) == 0:
            logger.warning('no keypoints found in file {} '.format(f))
        for kp, desc in r:
            kp_list.append(kp)
            desc_list.append(desc)
            fn_list.append(f)
    results = None

    logger.info('calculating pca (number of patches: {})'.format(len(desc_list)))
    from sklearn.decomposition import PCA

    pca = PCA(32, whiten=True)
    desc = pca.fit_transform(np.array(desc_list))
    desc_list = None

    logger.info('calculating new centers (shape: {})'.format(desc.shape))

    logger.info("starting KMeans (centers: {})".format(args.number_of_clusters))
    import sklearn.cluster

    kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=args.number_of_clusters, compute_labels=False,
                                             batch_size=10000 if args.number_of_clusters < 1000 else 50000)

    if desc.shape[0] > 500000:
        idx = np.linspace(0, desc.shape[0] - 1, 500000, dtype=np.int32)
        kmeans.fit(desc[idx])
    else:
        kmeans.fit(desc)

    center_path = os.path.join(args.out_dir[0], 'centers.pkl')
    logger.info('saving centers to %s' % center_path)
    pickle.dump(kmeans, open(center_path, 'wb'))

    patches_files = {}
    feature_count = 0
    batch_size = 50000
    for batch_start in tqdm(range(0, desc.shape[0], batch_size), 'Transforming and filtering'):
        def batch(d):
            return d[batch_start:batch_start + batch_size]


        b = batch(desc)
        dist = kmeans.transform(b)
        prediction = kmeans.predict(b)

        dist = np.sort(dist)
        ratio = dist[:, 0] / dist[:, 1]

        for p, f, c, r in zip(batch(kp_list), batch(fn_list), prediction, ratio):
            if r <= 0.9:
                feature_count += 1
                if f in patches_files:
                    patches_files[f].append((p, c))
                else:
                    patches_files[f] = [(p, c)]

    print('copying %i (all patches per page) image patches to %s ' % (feature_count, args.out_dir[0]))
    patches_results = Parallel(n_jobs=num_cores, verbose=9)(
        delayed(extract_patches)(filename, tup, args) for filename, tup in patches_files.items())

    print(f"type of results {type(patches_results)}")
    print(f"len(results) = {len(patches_results)}")

    # Global dictionary to hold consolidated patches by cluster
    all_clusters = defaultdict(list)
    # Consolidate patches from all images into the global cluster dictionary
    nnnn = 0
    for image_patches in patches_results:
        for cluster_id, patches in image_patches.items():
            all_clusters[cluster_id].extend(patches)
        print(f"{len(image_patches)} added to cluster {cluster_id}")# Append patches to the respective cluster
        print(f"{nnnn} iteration of results")
        nnnn +=1

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
