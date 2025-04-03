import cv2
import numpy as np
import matplotlib.pyplot as plt
from homography_manual import find_homography_manual, compute_transformed_coordinates, warp_perspective_manual

# =========================================================================================================
# TODO: you can add other functions here or files in this directory
# =========================================================================================================

src_points = np.array([
    [655, 268],  # 左上
    [995, 332],  # 右上
    [848, 342],  # 右下
    [535, 278]   # 左下
], dtype=np.float32)

# src_points = np.array([
# [656,270],[990, 332],[848,344],[535,279]
# ], dtype=np.float32)

dst_points = np.array([
    [2000, 1000],    # 左上（Y=5.5）
    [3832, 1000],    # 右上
    [3832, 1550],    # 右下（Y=0）
    [2000, 1550]     # 左下
], dtype=np.float32)

def calculate_distance(img):
    """
    Calculate the required distance through homography (metric: meter).
    Error less than 5% is ok.
    :param img: float/int array, shape: (height, width, channel)
    :return (distance 1, distance 2, distance 3), type: float
    """
    # =========================================================================================================
    # TODO: Please fill this part with your code
    # But DO NOT change this interface
    #H, _ = cv2.findHomography(src_points, dst_points)
    H = find_homography_manual(src_points, dst_points)

    width_px = int(5000)  # 18.32米 → 1832像素
    height_px = int(5000)   # 5.5米 → 550像素

    # This speed is too slow than cv2.warpPerspective--------:(
    img_transformed = warp_perspective_manual(img, H, (width_px, height_px))


    plt.figure(figsize=(50, 50))  # Use figsize parameter to specify dimensions
    plt.imshow(img_transformed)
    plt.show()
    
    img_transformed = cv2.cvtColor(img_transformed, cv2.COLOR_RGB2BGR)
    cv2.imwrite("src/problem3_homography/transformed_result.png", img_transformed)
    cv2.imwrite("transformed_result.png", img_transformed)

		# find where the [808, 300] point is located in the transformed image
    #gate_point = np.array([[811, 299]], dtype=np.float32)
    ball_point = np.array([[250, 360]], dtype=np.float32)
    referee_point = np.array([[20, 375]], dtype=np.float32)
    left_foot_point = np.array([[785, 295]], dtype=np.float32)
    gate_post_point = np.array([[875, 310]], dtype=np.float32)
    #gate_point_transformed = compute_transformed_coordinates(H, gate_point)
    ball_point_transformed = compute_transformed_coordinates(H, ball_point)
    referee_point_transformed = compute_transformed_coordinates(H, referee_point)
    left_foot_transformed = compute_transformed_coordinates(H, left_foot_point)
    gate_post_transformed = compute_transformed_coordinates(H, gate_post_point)
    #print(gate_point, 'gate point->', gate_point_transformed)
    print(ball_point, 'ball point->', ball_point_transformed)
    print(referee_point, 'referee point->', referee_point_transformed)

    distance_1 = ball_point_transformed[0][1] - gate_post_transformed[0][1]
    distance_2 = np.sqrt((left_foot_transformed[0][0] - gate_post_transformed[0][0]) ** 2 + 
                     (left_foot_transformed[0][1] - gate_post_transformed[0][1]) ** 2)
    distance_3 = np.sqrt((ball_point_transformed[0][0] - referee_point_transformed[0][0]) ** 2 + 
                     (ball_point_transformed[0][1] - referee_point_transformed[0][1]) ** 2)
    pass
    # =========================================================================================================

    return distance_1 / 100.0, distance_2 / 100.0, distance_3 / 100.0


if __name__ == '__main__':
    img_path = 'src/problem3_homography/football.png'

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # =========================================================================================================
    # for testing this demo:
    distance_1, distance_2, distance_3 = calculate_distance(img)
    print('distance_1 ball to gate: (meters)', distance_1)
    print('distance_2 left foot to gatepost: (meters)', distance_2)
    print('distance_3 referee to ball: (meters)', distance_3)
    
    # =========================================================================================================