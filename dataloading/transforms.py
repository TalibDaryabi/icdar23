import torch
import torch.nn.init
import torchvision.transforms as transforms
import numpy as np
import math
import random
import ocrodeg

normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


class PILTransforms:
    mean_image = None

    def get(self, transform):
        t = getattr(self, str(transform), None)
        assert t is not None, 'transform {} not known'.format(transform)
        return t()

    @staticmethod
    def train_transforms():
        transform = transforms.Compose([
            ResizeAndPad(),
            normalize
        ])
        return transform

    @staticmethod
    def val_transforms():
        transform = transforms.Compose([
            ResizeAndPad(),
            normalize
        ])
        return transform

    @staticmethod
    def test_transforms():
        transform = transforms.Compose([
            ResizeAndPad(),
            normalize
        ])
        return transform


class ResizeAndPad:

    def __init__(self, height=32, width=128, pad=False):
        self.height = height
        self.width = width
        self.pad = pad

    def __call__(self, img):

        if not self.pad:
            t1 = transforms.Resize((self.height, self.width))
            return t1(img)

        w_ = img.size[0] / self.width
        h_ = img.size[1] / self.height

        if w_ > h_:
            a = w_
        else:
            a = h_

        h, w = int(img.size[1] / a), int(img.size[0] / a)

        t1 = transforms.Resize((h, w))

        pad_w = (self.width - w)
        pad_h = (self.height - h)
        t2 = transforms.Pad((math.ceil(pad_w / 2), math.ceil(pad_h / 2), math.floor(pad_w / 2), math.floor(pad_h / 2)),
                            fill=255)
        return t2(t1(img))


# 1. Random Geometric Transformations
class Rotate:
    def __init__(self, angle=0.0):
        self.angle = angle

    def __call__(self, img):
        img_np = img.squeeze().numpy()
        transformed = ocrodeg.transform_image(img_np, angle=self.angle)
        return torch.tensor(transformed, dtype=torch.float32).unsqueeze(0)


class Scale:
    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, img):
        img_np = img.squeeze().numpy()
        transformed = ocrodeg.transform_image(img_np, scale=self.scale)
        return torch.tensor(transformed, dtype=torch.float32).unsqueeze(0)


class AnisotropicScale:
    def __init__(self, aniso=1.0):
        self.aniso = aniso

    def __call__(self, img):
        img_np = img.squeeze().numpy()
        transformed = ocrodeg.transform_image(img_np, aniso=self.aniso)
        return torch.tensor(transformed, dtype=torch.float32).unsqueeze(0)


class Translate:
    def __init__(self, translation=(0, 0)):
        self.translation = translation

    def __call__(self, img):
        img_np = img.squeeze().numpy()
        transformed = ocrodeg.transform_image(img_np, translation=self.translation)
        return torch.tensor(transformed, dtype=torch.float32).unsqueeze(0)


# 2. Random Distortions
class BoundedGaussianNoise:
    def __init__(self, sigma=1.0, maxdelta=5.0):
        self.sigma = sigma
        self.maxdelta = maxdelta

    def __call__(self, img):
        img_np = img.squeeze().numpy()
        noise = ocrodeg.bounded_gaussian_noise(img_np.shape, self.sigma, self.maxdelta)
        distorted = ocrodeg.distort_with_noise(img_np, noise)
        return torch.tensor(distorted, dtype=torch.float32).unsqueeze(0)


class NoiseDistort1D:
    def __init__(self, sigma=100.0, magnitude=100.0):
        self.sigma = sigma
        self.magnitude = magnitude

    def __call__(self, img):
        img_np = img.squeeze().numpy()
        deltas = ocrodeg.noise_distort1d(img_np.shape, self.sigma, self.magnitude)
        distorted = ocrodeg.distort_with_noise(img_np, deltas)
        return torch.tensor(distorted, dtype=torch.float32).unsqueeze(0)


# 3. Blur and Thresholding

class BinaryBlur:
    def __init__(self, blur_radius=2.0, noise=0.0):
        self.blur_radius = blur_radius
        self.noise = noise

    def __call__(self, img):
        # Convert PyTorch tensor to numpy array
        img_np = img.squeeze().numpy()
        # Apply ocrodeg binary_blur
        blurred = ocrodeg.binary_blur(img_np, self.blur_radius, noise=self.noise)
        # Convert back to PyTorch tensor
        return torch.tensor(blurred, dtype=torch.float32).unsqueeze(0)


class GaussianBlur:
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def __call__(self, img):
        img_np = img.squeeze().numpy()
        blurred = ndi.gaussian_filter(img_np, self.sigma)
        return torch.tensor(blurred, dtype=torch.float32).unsqueeze(0)


# 4. Multiscale Noise
class MultiscaleNoise:
    def __init__(self, scales, weights=None, limits=(0.0, 1.0)):
        self.scales = scales
        self.weights = weights
        self.limits = limits

    def __call__(self, img):
        noise = ocrodeg.make_multiscale_noise(img.shape, self.scales, self.weights, self.limits)
        return torch.tensor(noise, dtype=torch.float32).unsqueeze(0)


# 5. Random Blobs and Blotches

class RandomBlobs:
    def __init__(self, blobdensity=0.1, size=10, roughness=2.0):
        self.blobdensity = blobdensity
        self.size = size
        self.roughness = roughness

    def __call__(self, img):
        blobs = ocrodeg.random_blobs(img.shape, self.blobdensity, self.size, self.roughness)
        return torch.tensor(blobs, dtype=torch.float32).unsqueeze(0)


class RandomBlotches:
    # Random Blotches
    def __init__(self, fgblobs=0.1, bgblobs=0.1, fgscale=10, bgscale=10):
        self.fgblobs = fgblobs
        self.bgblobs = bgblobs
        self.fgscale = fgscale
        self.bgscale = bgscale

    def __call__(self, img):
        img_np = img.squeeze().numpy()
        blotched = ocrodeg.random_blotches(img_np, self.fgblobs, self.bgblobs, self.fgscale, self.bgscale)
        return torch.tensor(blotched, dtype=torch.float32).unsqueeze(0)


# 6. Fibrous Noise
class FibrousImage:
    def __init__(self, nfibers=300, l=300, a=0.2, stepsize=0.5, limits=(0.1, 1.0), blur=1.0):
        self.nfibers = nfibers
        self.l = l
        self.a = a
        self.stepsize = stepsize
        self.limits = limits
        self.blur = blur

    def __call__(self, img):
        fibrous = ocrodeg.make_fibrous_image(img.shape, self.nfibers, self.l, self.a, self.stepsize, self.limits,
                                             self.blur)
        return torch.tensor(fibrous, dtype=torch.float32).unsqueeze(0)


class PrintlikeMultiscale:
    def __init__(self, scales, weights=None, limits=(0.0, 1.0)):
        self.scales = scales
        self.weights = weights
        self.limits = limits

    def __call__(self, img):
        img_np = img.squeeze().numpy()
        degraded = ocrodeg.printlike_multiscale(img_np, self.scales, self.weights, self.limits)
        return torch.tensor(degraded, dtype=torch.float32).unsqueeze(0)


class PrintlikeFibrous:
    def __init__(self, nfibers=300, l=300, a=0.2, stepsize=0.5, limits=(0.1, 1.0), blur=1.0):
        self.nfibers = nfibers
        self.l = l
        self.a = a
        self.stepsize = stepsize
        self.limits = limits
        self.blur = blur

    def __call__(self, img):
        img_np = img.squeeze().numpy()
        degraded = ocrodeg.printlike_fibrous(img_np, self.nfibers, self.l, self.a, self.stepsize, self.limits,
                                             self.blur)
        return torch.tensor(degraded, dtype=torch.float32).unsqueeze(0)


# 8. Binary Erosion
class BinaryErode:
    def __init__(self, iterations=1):
        self.iterations = iterations

    def __call__(self, img):
        img_np = img.squeeze().numpy()
        eroded = ocrodeg.binary_erode(img_np, iterations=self.iterations)
        return torch.tensor(eroded, dtype=torch.float32).unsqueeze(0)


# 9. Binary Closing
class BinaryClose:
    def __init__(self, iterations=1):
        self.iterations = iterations

    def __call__(self, img):
        img_np = img.squeeze().numpy()
        closed = ocrodeg.binary_close(img_np, iterations=self.iterations)
        return torch.tensor(closed, dtype=torch.float32).unsqueeze(0)


# 10. Binary Opening
class BinaryOpen:
    def __init__(self, iterations=1):
        self.iterations = iterations

    def __call__(self, img):
        img_np = img.squeeze().numpy()
        opened = ocrodeg.binary_open(img_np, iterations=self.iterations)
        return torch.tensor(opened, dtype=torch.float32).unsqueeze(0)
