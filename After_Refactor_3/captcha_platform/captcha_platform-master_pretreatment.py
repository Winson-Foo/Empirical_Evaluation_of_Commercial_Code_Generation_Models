import cv2


class ImagePreprocessor:
    def __init__(self, image):
        self.image = image

    def get_image(self):
        return self.image

    def binarize(self, threshold, modify=False):
        ret, binarized_image = cv2.threshold(self.image, threshold, 255, cv2.THRESH_BINARY)
        if modify:
            self.image = binarized_image
        return binarized_image


def preprocess_image(image, threshold=-1):
    preprocessor = ImagePreprocessor(image)
    if threshold > 0:
        preprocessor.binarize(threshold, modify=True)
    return preprocessor.get_image()


def preprocess_image_by_func(exec_map, key, src_image):
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