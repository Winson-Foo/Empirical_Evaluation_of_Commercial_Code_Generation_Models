import tensorflow.keras.backend as K

MAX_PIXEL_VALUE = 1
Y_WEIGHTS = [65.738, 129.057, 25.064]
Y_OFFSET = 16


def peak_signal_to_noise_ratio(y_true, y_pred, max_pixel_value=MAX_PIXEL_VALUE):
    """
    Computes the peak signal-to-noise ratio (PSNR) between two images.

    Args:
        y_true: the ground truth image.
        y_pred: the predicted image.
        max_pixel_value: the maximum value of the pixel range (default=1).
    
    Returns:
        The PSNR value between the two images.
    """
    mse = K.mean(K.square(y_pred - y_true))
    psnr = -10.0 * K.log(mse) / K.log(10.0)
    return 20 * K.log(max_pixel_value) / K.log(10.0) - psnr


def convert_rgb_to_y(image):
    """
    Converts an RGB image to the Y channel.

    Args:
        image: the RGB image.

    Returns:
        The Y channel of the image.
    """
    r = image[:, :, :, 0]
    g = image[:, :, :, 1]
    b = image[:, :, :, 2]

    y = Y_OFFSET + Y_WEIGHTS[0] * r + Y_WEIGHTS[1] * g + Y_WEIGHTS[2] * b
    return y / 255.0


def peak_signal_to_noise_ratio_y(y_true, y_pred, max_pixel_value=MAX_PIXEL_VALUE):
    """
    Computes the peak signal-to-noise ratio (PSNR) on the Y channel of an image.

    Args:
        y_true: the ground truth image.
        y_pred: the predicted image.
        max_pixel_value: the maximum value of the pixel range (default=1).
    
    Returns:
        The PSNR value on the Y channel of the two images.
    """
    y_true_y = convert_rgb_to_y(y_true)
    y_pred_y = convert_rgb_to_y(y_pred)
    return peak_signal_to_noise_ratio(y_true_y, y_pred_y, max_pixel_value)