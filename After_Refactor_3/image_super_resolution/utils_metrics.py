import tensorflow.keras.backend as K

def calculate_mse(y_true, y_pred):
    """
    Calculates the mean squared error (MSE) between y_true and y_pred.
    Args:
        y_true: ground truth.
        y_pred: predicted value.    
    Returns:
        The MSE value.
    """
    return K.mean(K.square(y_pred - y_true))


def calculate_psnr(mse, max_pixel):
    """
    Calculates the peak signal-to-noise ratio (PSNR) value.
    Args:
        mse: mean squared error (MSE) value.
        max_pixel: maximum value of the pixel range (default=1).
    Returns:
        The PSNR value.
    """
    return -10.0 * K.log(mse) / K.log(10.0) + 20.0 * K.log(max_pixel) / K.log(10.0)

def PSNR(y_true, y_pred, MAXp=1):
    """
    Evaluates the PSNR value:
        PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
    Args:
        y_true: ground truth.
        y_pred: predicted value.
        MAXp: maximum value of the pixel range (default=1).
    Returns:
        The PSNR value.
    """
    mse = calculate_mse(y_true, y_pred)
    return calculate_psnr(mse, MAXp)


def RGB_to_Y(image):
    """
    Converts RGB image to Y channel.
    Args:
        image: A tensor representing the RGB image with values from 0 to 1.
    Returns:
        A tensor representing the Y channel of the image.
    """
    R = image[:, :, :, 0]
    G = image[:, :, :, 1]
    B = image[:, :, :, 2]
    
    Y = 16 + (65.738 * R) + 129.057 * G + 25.064 * B
    return Y / 255.0


def PSNR_Y(y_true, y_pred, MAXp=1):
    """
    Evaluates the PSNR value on the Y channel.
    Args:
        y_true: ground truth.
        y_pred: predicted value.
        MAXp: maximum value of the pixel range (default=1).
    Returns:
        The PSNR value.
    """
    y_true = RGB_to_Y(y_true)
    y_pred = RGB_to_Y(y_pred)
    mse = calculate_mse(y_true, y_pred)
    return calculate_psnr(mse, MAXp)