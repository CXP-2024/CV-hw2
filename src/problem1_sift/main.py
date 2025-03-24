import cv2 


class SIFT(object):
    def __init__(self, **kwargs):
        """
        Implement Scale-Invariant Feature Transform(SIFT) algorithm
        :param kwargs: other hyperparameters, such as sigma, blur ratio, border, etc.
        """
        pass

    # =========================================================================================================
    # TODO: you can add other functions here or files in this directory
    # =========================================================================================================

    def out(self, img):
        """
        Implement Scale-Invariant Feature Transform(SIFT) algorithm
        :param img: float/int array, shape: (height, width, channel)
        :return sift_results (keypoints, descriptors)
        """
        # =========================================================================================================
        # TODO: Please fill this part with your code
        # But DO NOT change this interface
        keypoints = None
        descriptors = None
        pass
        # =========================================================================================================

        return keypoints, descriptors

    def vis(self, img):
        """
        Visualize the key points of the given image, you can save the result as an image or just plot it.
        :param img: float/int array, shape: (height, width, channel)
        :return your own stuff (DIY is ok)
        """
        # =========================================================================================================
        # TODO: Please fill this part with your code
        # But DO NOT change this interface
        # =========================================================================
				# 
				# 
				# ================================
        pass

    def match(self, img1, img2):
        """
        Match keypoints between img1 and img2 and draw lines between the corresponding keypoints;
        you can save the result as an image or just plot it.
        :param img1: float/int array, shape: (height, width, channel)
        :param img1: float/int array, shape: (height, width, channel)
        :return your own stuff (DIY is ok)
        """
        # =========================================================================================================
        # TODO: Please fill this part with your code
        # But DO NOT change this interface
        # =========================================================================================================
        pass


if __name__ == '__main__':
    img_path = 'school_gate.jpeg'

    kwargs = {}

    # =========================================================================================================
    # for testing this demo:
    img = cv2.imread(img_path)
    sift = SIFT(**kwargs)
    sift.vis(img)
    sift.match(img, img)
    # =========================================================================================================
