import cv2
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import os
import glob
import sklearn.preprocessing
from sklearn.decomposition import PCA
from tqdm import tqdm
import pickle
import bz2

from utils.logging_script import setup_logging

script_name = os.path.basename(__file__)
print(f"Running script: {script_name}")
logger = setup_logging(script_name)

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

    sift = cv2.SIFT_create(sigma=args.sigma)
    kp = sift.detect(img_gray, None)
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
            new_kp.append(p)
            uniq_kp.append(point)

    _, desc = sift.compute(img_gray, new_kp)

    desc = sklearn.preprocessing.normalize(desc, norm='l1')
    desc = np.sign(desc) * np.sqrt(np.abs(desc))
    desc = sklearn.preprocessing.normalize(desc, norm='l2')

    print(f'{len(new_kp)} out of {len(kp)} for {os.path.splitext(os.path.basename(input_img))[0]}')
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
    if len(tup) > args.patches_per_page and args.patches_per_page != -1:
        idx = np.linspace(0, len(tup) - 1, args.patches_per_page, dtype=np.int32)
        tup = [tup[i] for i in idx]

    points, _ = zip(*tup)

    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if args.scale != -1:
        scale = args.scale
        height = img.shape[0] * scale
        width = img.shape[1] * scale
        new_size_mask = (int(width), int(height))
        img = cv2.resize(img, new_size_mask, interpolation=cv2.INTER_CUBIC)

    # Create a list to hold all patches
    patch_data = []
    count = 0
    for p in points:
        roi = get_image_patch(img, p[0], p[1], args)
        if roi is None:
            continue

        # Store patch in memory instead of writing to file
        patch_data.append({  # Original image name
            "patch": roi,  # Patch data (numpy array)
            "keypoint": p  # Keypoint coordinates
        })
        count = count + 1

    return os.path.basename(filename), patch_data  # Return filename and its patches


def gather_image_files_with_oswalk(directory, extensions):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                yield os.path.join(root, file)


if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser(description="extract patches from images")
    parser.add_argument('--in_dir', metavar='in_dir', dest='in_dir', type=str, nargs=1,
                        help='input directory', required=True)
    parser.add_argument('--out_dir', metavar='out_dir', dest='out_dir', type=str, nargs=1,
                        help='output directory', required=True)
    parser.add_argument('--win_size', metavar='win_size', dest='win_size', type=int, nargs='?',
                        help='size of the patch',
                        default=32)
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

    if not os.path.exists(args.out_dir[0]):
        logger.info('creating directory %s' % args.out_dir[0])
        os.mkdir(args.out_dir[0])

    assert len(os.listdir(args.out_dir[0])) == 0, 'out_dir is not empty'

    assert args.win_size % 2 == 0, 'win_size must be even'

    num_cores = int(multiprocessing.cpu_count() / 2)
    num_cores = 14
    path_to_centers = ''

    # Gather all image files
    files = list(gather_image_files_with_oswalk(args.in_dir[0], IMG_EXTENSIONS))

    assert len(files) > 0, 'no images found'
    logger.info('Found {} images'.format(len(files)))

    logger.info('calculating features for images in %s (number of cores:%d)' % (args.in_dir, num_cores))
    results = []
    results = Parallel(n_jobs=num_cores, verbose=9)(delayed(calc_features)(f, args) for f in files)

    logger.info('collecting descriptors')
    desc_list = []
    kp_list = []
    fn_list = []

    for r, f in tqdm(zip(results, files)):
        if len(r) == 0:
            logger.warning('no keypoints found in file {} '.format(f))
        for kp, desc in r:
            kp_list.append(kp)
            desc_list.append(desc)
            fn_list.append(f)

    # logger.info('copying %i (all patches per page) image patches to %s ' % (kp_list, args.out_dir[0]))
    all_patches = {}
    patches_results = Parallel(n_jobs=num_cores, verbose=9)(
        delayed(extract_patches)(filename, tup, args) for filename, tup in zip(files, results))
    # Consolidate results into a single dictionary
    for filename, patches in patches_results:
        all_patches[filename] = patches

        # Save the consolidated dictionary to a single file
    output_file = os.path.join(args.out_dir[0], "patches.pkl.bz2")
    with bz2.BZ2File(output_file, "wb") as f:
        pickle.dump(all_patches, f)

    config_out_path = os.path.join(args.out_dir[0], 'db-creation-parameters.json')
    logger.info('writing config parameters to {}')

    with open(config_out_path, 'w') as f:
        import json
        json.dump(vars(args), f)
