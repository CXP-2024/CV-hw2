import cv2 
from problem1_sift.main import SIFT


class Stitch(SIFT):
    def __init__(self, **kwargs):
        """
        Implement panorama stitching with RANSAC
        :param kwargs: other hyperparameters of SIFT, such as sigma, blur ratio, border, etc.
        """
        super().__init__(**kwargs)
        pass

    # =========================================================================================================
    # TODO: you can add other functions here or files in this directory
    # =========================================================================================================

    def stitch_img_lst(self, img_lst):
        """
        Stitch a list of image in order (NOT RANDOM),
        you can save the result as an image or just plot it.
        :param img_lst: a list of images, the shape of image is (height, width, channel)
        :return your own stuff (DIY is ok)
        """
        # =========================================================================================================
        # TODO: Please fill this part with your code
        # But DO NOT change this interface
        # =========================================================================================================
        pass


if __name__ == '__main__':
    path = 'building_image'
    img_lst = []

    # =========================================================================================================
    # for testing this demo:
    stitch = Stitch()
    stitch.stitch_img_lst(img_lst)
    # =========================================================================================================
