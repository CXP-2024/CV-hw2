import cv2 
from sift_useful import SIFT, ransac_match


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
        keypoints1, descriptors1 = self.out(img1, contrast_threshold=0.05)
        keypoints2, descriptors2 = self.out(img2, contrast_threshold=0.6)
        matches = self.match_descriptors_(descriptors1, descriptors2, ratio_threshold=0.8)
        print('matches:', len(matches))
        self.visualize_matches_(img1, keypoints1, img2, keypoints2, matches)
        
				########### DO the ransac match here ############
        best_inliers, best_homography = self.ransac_match_(keypoints1, keypoints2, matches)
        print('after ransac best_inliers:', len(best_inliers))
        self.visualize_matches_(img1, keypoints1, img2, keypoints2, best_inliers)
        # =========================================================================================================
    
    def ransac_match_(self, keypoints1, keypoints2, matches, num_iterations=1000, inlier_threshold=7):
        return ransac_match(keypoints1, keypoints2, matches, num_iterations, inlier_threshold)


if __name__ == '__main__':
    img1_path = 'src/problem2_match/book_reference.jpeg'
    img2_path = 'src/problem2_match/books.jpeg'

    kwargs = {}

    # =========================================================================================================
    # for testing this demo:
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    # change to rgb
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    r_sift = R_SIFT(**kwargs)
    r_sift.match(img1, img2)
    # =========================================================================================================
