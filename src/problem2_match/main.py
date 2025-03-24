import cv2 
from problem1_sift.main import SIFT


class R_SIFT(SIFT):
    def __init__(self, **kwargs):
        """
        Implement Scale-Invariant Feature Transform(SIFT) algorithm with RANSAC
        :param kwargs: other hyperparameters, such as sigma, blur ratio, border, etc.
        """
        super().__init__(**kwargs)
        pass

    # =========================================================================================================
    # TODO: you can add other functions here or files in this directory
    # =========================================================================================================

    def match(self, img1, img2):
        """
        Match keypoints between img1 and img2 and draw lines between the corresponding keypoints with RANSAC,
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
    img1_path = 'book_reference.jpeg'
    img2_path = 'books.jpeg'

    kwargs = {}

    # =========================================================================================================
    # for testing this demo:
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    r_sift = R_SIFT(**kwargs)
    r_sift.match(img1, img2)
    # =========================================================================================================
