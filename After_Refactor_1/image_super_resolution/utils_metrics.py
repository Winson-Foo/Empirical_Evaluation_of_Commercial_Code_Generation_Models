import tensorflow.keras.backend as K


def calculate_psnr(ground_truth, predicted_value, max_pixel_value=1):
    """
    Calculates the PSNR value:
        PSNR = 20 * log10(MAXp) - 10 * log10(MSE).

    Args:
        ground_truth: Ground truth.
        predicted_value: Predicted value.
        max_pixel_value: Maximum value of the pixel range (default=1).
    """
    psnr = -10.0 * K.log(K.mean(K.square(predicted_value - ground_truth))) / K.log(10.0)
    return psnr


def convert_rgb_to_y_channel(image):
    """
    Converts an RGB image to the Y channel.
    
    Args:
        image: RGB image with values from 0 to 1.
        
    Returns:
        Y channel with values from 0 to 1.
    """
    r, g, b = image[:, :, :, 0], image[:, :, :, 1], image[:, :, :, 2]
    y = 16 + (65.738 * r) + 129.057 * g + 25.064 * b
    return y / 255.0


def calculate_psnr_on_y_channel(ground_truth, predicted_value, max_pixel_value=1):
    """
    Calculates the PSNR value on the Y channel:
        PSNR = 20 * log10(MAXp) - 10 * log10(MSE).

    Args:
        ground_truth: Ground truth.
        predicted_value: Predicted value.
        max_pixel_value: Maximum value of the pixel range (default=1).
    """
    y_ground_truth = convert_rgb_to_y_channel(ground_truth)
    y_predicted_value = convert_rgb_to_y_channel(predicted_value)
    psnr = -10.0 * K.log(K.mean(K.square(y_predicted_value - y_ground_truth))) / K.log(10.0)
    return psnr