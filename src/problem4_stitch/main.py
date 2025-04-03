import cv2 
from sift_useful import SIFT


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
        # Detect and match features
        img1 = img_lst[0]
        img2 = img_lst[1]
        img3 = img_lst[2]
        img4 = img_lst[3]
        # compute matches
        keypoints1, keypoints2, matches = self.detect_and_match_features_(img1, img2, num_octaves=4, scales_per_octave=4, contrast_threshold=0.735, edge_threshold=10, ratio_threshold=0.7)
        keypoints4, keypoints3,  matches2 = self.detect_and_match_features_(img4, img3, num_octaves=4, scales_per_octave=4, contrast_threshold=0.7, edge_threshold=10, ratio_threshold=0.7)
        
				# compute homography and combine 1 and 2, 3 and 4
        ransac_matches, homography = self.compare_ransac_matching_(img1, img2, keypoints1,  keypoints2, matches)
        ransac_matches2, homography2 = self.compare_ransac_matching_(img4, img3, keypoints4,  keypoints3, matches2)
        warped_img1, img2_canvas, blended, overlap_mask = self.warp_image_(img1, img2, homography)
        img4_canvas, warped_img3, blended2, overlap_mask2 = self.warp_image_(img4, img3, homography2)
        
				# Stitch the images: blend + blend2
        img1_gray = blended
        img2_gray = blended2
        keypoints1, keypoints2, matches = self.detect_and_match_features_(img1_gray, img2_gray, num_octaves=4, scales_per_octave=4, contrast_threshold=0.65, edge_threshold=10, ratio_threshold=0.75)
        ransac_matches, homography = self.compare_ransac_matching_(img1_gray, img2_gray, keypoints1, keypoints2, matches)
        warped_img1, img2_canvas, blended, overlap_mask = self.warp_image_(img1_gray, img2_gray, homography)
        # =========================================================================================================
        return blended


if __name__ == '__main__':
    image_path1 = "src/problem4_stitch/building_image/building_1.jpg"  # Replace with your image path
    image_path2 = "src/problem4_stitch/building_image/building_2.jpg"  # Replace with your image path
    image_path3 = "src/problem4_stitch/building_image/building_3.jpg"  # Replace with your image path
    image_path4 = "src/problem4_stitch/building_image/building_4.jpg"  # Replace with your image path
    img_lst = []
    img1 = cv2.imread(image_path1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img_lst.append(img1)
    img2 = cv2.imread(image_path2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img_lst.append(img2)
    img3 = cv2.imread(image_path3)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    img_lst.append(img3)
    img4 = cv2.imread(image_path4)
    img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
    img_lst.append(img4)

    # =========================================================================================================
    # for testing this demo:
    stitch = Stitch()
    blend = stitch.stitch_img_lst(img_lst)
    # save the result
    cv2.imwrite('src/problem4_stitch/building_panorama.jpg', cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))
    # =========================================================================================================
