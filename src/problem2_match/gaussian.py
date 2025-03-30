import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import tqdm

def my_conv2d(kernel, img, conv_type="full"):
    """
    Implement 2D image convolution
    :param kernel: float/int array, shape: (x, x)
    :param img: float/int array, shape: (height, width)
    :param conv_type: str, convolution padding choices, should in ['full', 'same', 'valid']
    :return conv results, numpy array
    """
    k_h, k_w = kernel.shape
    i_h, i_w = img.shape

    kernel = np.rot90(kernel, 2)

    # Calculate output dimensions based on conv_type
    half_k_h = k_h // 2
    half_k_w = k_w // 2
    if conv_type == "full":
        pad_h = k_h - 1
        pad_w = k_w - 1
        out_h = i_h + pad_h
        out_w = i_w + pad_w
    elif conv_type == "same":
        pad_h = k_h // 2
        pad_w = k_w // 2
        out_h = i_h
        out_w = i_w
    elif conv_type == "valid":
        pad_h = pad_w = 0
        out_h = i_h - k_h + 1
        out_w = i_w - k_w + 1
    else:
        raise ValueError("conv_type must be 'full', 'same', or 'valid'")

    # Pad the image, put it in the center with 0 outside
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    # Perform convolution
    result = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            img_region = padded_img[i : i + k_h, j : j + k_w]
            result[i, j] = np.sum(img_region * kernel)
            
    pass
    # =========================================================================================================
    return result

def my_conv3d(kernel, img, conv_type="full"):
    """
    Implement 3D image convolution
    :param kernel: float/int array, shape: (ci, h, w, co)
    :param img: float/int array, shape: (height, width, channel)
    :param conv_type: str, convolution padding choices, should in ['full', 'same', 'valid']
    :return conv results, numpy array
    """
    assert len(kernel.shape) == 4 and len(img.shape) == 3, "The dimensions of kernel and img should be 3."
    ci, k_h, k_w, co = kernel.shape
    i_h, i_w, i_c = img.shape

    result = np.zeros((i_h, i_w, i_c))
    for c_i in range(ci):
      for c_o in range(co):
        result[:, :, c_o] = my_conv2d(kernel[c_i, :, :, c_o], img[:, :, c_o], conv_type=conv_type)
        
    return result

def generate_gaussian_kernel(size, sigma = 1):
    '''This function generates a 2D Gaussian kernel. And it output a 2D numpy array whose values are the Gaussian values between 0 and 1.'''
    # generate an axis
    x = np.linspace(-size // 2, size // 2, size)
    gauss_x = np.exp(-x ** 2 / (2 * sigma ** 2))

    # generate a 2D gaussian kernel and normalize
    gauss_2d = np.outer(gauss_x, gauss_x)
    gauss_2d /= gauss_2d.sum()
    return gauss_2d

def gaussian_filter(img, kernel_size, sigma=1):
		"""
		Implement Gaussian filter
		:param img: float/int array, shape: (height, width)
		:param kernel_size: int, the size of the kernel, should be odd number
		:param sigma: float, standard deviation of the Gaussian distribution
		:return filtered image, numpy array
		"""
		assert kernel_size % 2 == 1, "kernel size should be odd number"
		
		# Generate Gaussian kernel
		kernel = generate_gaussian_kernel(kernel_size, sigma=sigma)

		# Perform convolution using my_conv2d function
		filtered_img = my_conv2d(kernel, img, conv_type="same")
		
		return filtered_img