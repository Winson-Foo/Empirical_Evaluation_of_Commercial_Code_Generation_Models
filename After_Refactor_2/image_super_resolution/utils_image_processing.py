import numpy as np


class ImageProcessor:
    def process(self, image):
        raise NotImplemented

    def unprocess(self, tensor):
        raise NotImplemented


class RGBImageProcessor(ImageProcessor):
    def process(self, image):
        """ Process a 3-dimensional RGB array into a scaled tensor of size (batch_size, height, width, channels). """
        tensor = image.astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)
        return tensor

    def unprocess(self, tensor):
        """ Transforms the 4-dimensional output tensor into a suitable RGB image format. """
        image = tensor.clip(0, 1) * 255
        image = np.uint8(image)
        return image


class PatchExtractor:
    def __init__(self, patch_size, padding_size):
        self.patch_size = patch_size
        self.padding_size = padding_size

    def extract(self, image):
        """ Splits the image into partially overlapping patches.

        The patches overlap by padding_size pixels.

        Pads the image twice:
            - first to have a size multiple of the patch size,
            - then to have equal padding at the borders.

        Args:
            image: numpy array of the input image.
        """
        xmax, ymax, _ = image.shape
        x_remainder = xmax % self.patch_size
        y_remainder = ymax % self.patch_size

        # modulo here is to avoid extending of patch_size instead of 0
        x_extend = (self.patch_size - x_remainder) % self.patch_size
        y_extend = (self.patch_size - y_remainder) % self.patch_size

        # make sure the image is divisible into regular patches
        extended_image = np.pad(image, ((0, x_extend), (0, y_extend), (0, 0)), 'edge')

        # add padding around the image to simplify computations
        padded_image = self.pad_patch(extended_image)

        xmax, ymax, _ = padded_image.shape
        patches = []

        x_lefts = range(self.padding_size, xmax - self.padding_size, self.patch_size)
        y_tops = range(self.padding_size, ymax - self.padding_size, self.patch_size)

        for x in x_lefts:
            for y in y_tops:
                x_left = x - self.padding_size
                y_top = y - self.padding_size
                x_right = x + self.patch_size + self.padding_size
                y_bottom = y + self.patch_size + self.padding_size
                patch = padded_image[x_left:x_right, y_top:y_bottom, :]
                patches.append(patch)

        return np.array(patches), padded_image.shape

    def pad_patch(self, image_patch):
        """ Pads image_patch with with padding_size edge values. """
        return np.pad(
            image_patch,
            ((self.padding_size, self.padding_size), (self.padding_size, self.padding_size), (0, 0)),
            'edge',
        )

    def unpad_patches(self, image_patches):
        return image_patches[:, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size, :]


class PatchCombiner:
    def __init__(self, patch_size, padding_size):
        self.patch_size = patch_size
        self.padding_size = padding_size

    def combine(self, patches, padded_image_shape, target_shape):
        """ Reconstruct the image from overlapping patches.

        After scaling, shapes and padding should be scaled too.

        Args:
            patches: patches obtained with extract_patches_from_image
            padded_image_shape: shape of the padded image constructed in extract_patches_from_image
            target_shape: shape of the final image
        """
        xmax, ymax, _ = padded_image_shape
        patches = self.unpad_patches(patches)
        patch_size = patches.shape[1]
        n_patches_per_row = ymax // patch_size

        complete_image = np.zeros((xmax, ymax, 3))

        row = -1
        col = 0
        for i in range(len(patches)):
            if i % n_patches_per_row == 0:
                row += 1
                col = 0
            complete_image[
            row * patch_size: (row + 1) * patch_size, col * patch_size: (col + 1) * patch_size, :
            ] = patches[i]
            col += 1
        return complete_image[0: target_shape[0], 0: target_shape[1], :]


class ImageLoader:
    def load(self, path):
        raise NotImplemented


class NumpyLoader(ImageLoader):
    def load(self, path):
        return np.load(path)


class ImageTransformer:
    def transform(self, image, patch_size, padding_size):
        """ Transforms the input image by splitting it into patches, processing the patches,
        and combining the processed patches into the final image.

        Args:
            image: numpy array of the input image.
            patch_size: size of the patches from the original image (without padding).
            padding_size: size of the overlapping area.
        """
        processor = RGBImageProcessor()
        patches, padded_image_shape = PatchExtractor(patch_size, padding_size).extract(image)
        processed_patches = processor.process(patches)
        output_tensor = self.process_patches(processed_patches)
        output_patches = processor.unprocess(output_tensor)
        result = PatchCombiner(patch_size, padding_size).combine(output_patches, padded_image_shape, image.shape)
        return result

    def process_patches(self, patches):
        raise NotImplemented


class SuperResolutionTransformer(ImageTransformer):
    def __init__(self, model):
        self.model = model

    def process_patches(self, patches):
        return self.model.predict(patches)


def test_transformer(transformer, image_path, patch_size=64, padding_size=16):
    loader = NumpyLoader()
    image = loader.load(image_path)
    result = transformer.transform(image, patch_size, padding_size)
    assert (result.shape == image.shape)
    assert (np.min(result) >= 0 and np.max(result) <= 255)
    return result