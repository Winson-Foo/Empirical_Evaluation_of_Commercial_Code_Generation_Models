import cv2


class ImagePreprocessor:
    def __init__(self, image):
        self.image = image

    def get_image(self):
        """Returns the original image."""
        return self.image

    def binarization(self, value, modify=False):
        """Applies binarization to the image.

        Args:
            value (int): Threshold value for binarization.
            modify (bool): Whether to modify the image in-place.
        
        Returns:
            The binarized image.
        """
        _, binarized_image = cv2.threshold(self.image, value, 255, cv2.THRESH_BINARY)
        if modify:
            self.image = binarized_image
        return binarized_image


def preprocess_image(image, binarization=-1):
    """Preprocesses the image by applying binarization if specified.

    Args:
        image (ndarray): The input image.
        binarization (int): Threshold value for binarization. Default is -1 (no binarization).
    
    Returns:
        The preprocessed image.
    """
    preprocessor = ImagePreprocessor(image)
    if binarization > 0:
        preprocessor.binarization(binarization, modify=True)
    return preprocessor.get_image()


def preprocess_image_by_func(exec_map, key, src_image):
    """Preprocesses the image based on a function map.

    Args:
        exec_map (dict): The function map containing preprocessing instructions.
        key (str): The key to retrieve the preprocessing instructions from the map.
        src_image (ndarray): The input image.

    Returns:
        The preprocessed image.
    """
    if not exec_map:
        return src_image
    target_image = cv2.cvtColor(src_image, cv2.COLOR_RGB2BGR)
    for sentence in exec_map.get(key):
        if sentence.startswith("@@"):
            target_image = eval(sentence[2:])
        elif sentence.startswith("$$"):
            exec(sentence[2:])
    return cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    pass