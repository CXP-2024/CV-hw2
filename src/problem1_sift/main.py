import cv2 
from sift_useful import sift, match_descriptors, visualize_keypoints, visualize_matches
import numpy as np

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
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift(gray_image, num_octaves=4, scales_per_octave=4, contrast_threshold=0.6, edge_threshold=10)
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
        keypoints, descriptors = sift(img, num_octaves=4, scales_per_octave=4, contrast_threshold=0.6, edge_threshold=10)
        visualize_keypoints(img, keypoints)
				# ================================
        pass

    def match(self, img1, img2, contrast_threshold1=0.6, contrast_threshold2=0.6, ratio_threshold=0.7, edge_threshold=10, num_octaves=4, scales_per_octave=4):
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
        keypoints1, descriptors1 = sift(img1, num_octaves, scales_per_octave, contrast_threshold1, edge_threshold)
        keypoints2, descriptors2 = sift(img2, num_octaves, scales_per_octave, contrast_threshold2, edge_threshold)
        visualize_keypoints(img1, keypoints1)
        visualize_keypoints(img2, keypoints2)
        matches = match_descriptors(descriptors1, descriptors2, ratio_threshold)
        visualize_matches(img1, keypoints1, img2, keypoints2, matches)
        # =========================================================================================================
        
    def visualize_matches_(self, img1, keypoints1, img2, keypoints2, matches):
        return visualize_matches(img1, keypoints1, img2, keypoints2, matches)
    def visualize_keypoints_(self, image, keypoints):
        return visualize_keypoints(image, keypoints)
    def sift_(self, gray_image, num_octaves=4, scales_per_octave=4, contrast_threshold=0.6, edge_threshold=10):
        return sift(gray_image, num_octaves, scales_per_octave, contrast_threshold, edge_threshold)
    def match_descriptors_(self, desc1, desc2, ratio_threshold=0.75):
        return match_descriptors(desc1, desc2, ratio_threshold)


if __name__ == '__main__':
    img_path = 'src/problem1_sift/school_gate.jpeg'
    img1 = cv2.imread('src/problem1_sift/building_2.jpg')
    img2 = cv2.imread('src/problem1_sift/building_2r45.jpg')  # rotate 45 degree

    kwargs = {}
    sift_ = SIFT(**kwargs)

    # =========================================================================================================
    # for testing this demo:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
		# =========================================================================================================
    sift_.match(img1, img2, contrast_threshold1=0.75, contrast_threshold2=0.6, ratio_threshold=0.7, edge_threshold=10, num_octaves=4, scales_per_octave=4)


    ################    Test the school gate image    ################
    sift_.vis(img)
    sift_.match(img, img)
    # =========================================================================================================