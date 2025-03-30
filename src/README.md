## README
If you have questions, please contact the TA.

### Notice
You are required to implement your own code in this framework.

Fill the TODO part of main.py in each problem directory. 
You are free to add your own functions and files in each directory.

However, **DO NOT** change:
* The name of `main.py` files
* The name of defined interface functions
* The **return** types of defined interface functions

### Problem 1
Solve the `SIFT` problem in the `problem1_sift` directory.

This problem can be divided into several parts:
* Create a Gaussian Pyramid
* Create a DOG (Difference of Gaussian) Pyramid
* Detect keypoints
* Compute the dominant orientation
* Compute the descriptor (128 dimensions) for each keypoint
* Match the keypoints according to the descriptors
* Add visualization functions for keypoints and matches

### Problem 2
Solve the `Matching` problem in the `problem2_match` directory.

This part implements RANSAC on the matches obtained using SIFT from Problem 1:
* Randomly select 4 matches and compute the homography matrix
* Apply the matrix to the remaining matched points from the first image, then check the distance between the matched points in the second image and the transformed points. If the distance is small enough, consider it an inlier
* Repeat this process to find the homography that produces the largest number of inliers
* Return the corresponding inlier matches and the best homography matrix

### Problem 3
Solve the `Homography` problem in the `problem3_homography` directory.

This part implements a function that calculates the homography matrix from 4 pairs of corresponding points.

### Problem 4
Solve the `Panorama Stitching` problem in the `problem4_stitch` directory.

This part uses SIFT and RANSAC to obtain the homography matrix and then applies it to one of the images before blending. The panorama is created through a series of steps:
* Blend image1 and image2 → blend1
* Blend image3 and image4 → blend2
* Blend blend1 and blend2 → final panorama
