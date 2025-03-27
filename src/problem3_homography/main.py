import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================================================================================================
# TODO: you can add other functions here or files in this directory
# =========================================================================================================

src_points = np.array([
    [656, 271],  # 左上
    [987, 332],  # 右上
    [848, 344],  # 右下
    [538, 279]   # 左下
], dtype=np.float32)

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
    H, _ = cv2.findHomography(src_points, dst_points)

    width_px = int(5000)  # 18.32米 → 1832像素
    height_px = int(5000)   # 5.5米 → 550像素


    img_transformed = cv2.warpPerspective(
				img, 
				H, 
				(width_px, height_px), 
				flags=cv2.INTER_LINEAR
		)


    plt.figure(figsize=(50, 50))  # Use figsize parameter to specify dimensions
    plt.imshow(img_transformed)
    plt.show()
    
    img_transformed = cv2.cvtColor(img_transformed, cv2.COLOR_RGB2BGR)
    cv2.imwrite("src/problem3_homography/transformed_result.png", img_transformed)

		# find where the [808, 300] point is located in the transformed image
    gate_point = np.array([[808, 300]], dtype=np.float32)
    ball_point = np.array([[245, 361]], dtype=np.float32)
    referee_point = np.array([[18, 377]], dtype=np.float32)
    left_foot_point = np.array([[780, 294]], dtype=np.float32)
    gate_post_point = np.array([[876, 311]], dtype=np.float32)
    gate_point_transformed = cv2.perspectiveTransform(gate_point[None, :, :], H)
    ball_point_transformed = cv2.perspectiveTransform(ball_point[None, :, :], H)
    referee_point_transformed = cv2.perspectiveTransform(referee_point[None, :, :], H)
    left_foot_transformed = cv2.perspectiveTransform(left_foot_point[None, :, :], H)
    gate_post_transformed = cv2.perspectiveTransform(gate_post_point[None, :, :], H)
    print(gate_point, 'gate point->', gate_point_transformed)
    print(ball_point, 'ball point->', ball_point_transformed)
    print(referee_point, 'referee point->', referee_point_transformed)

    distance_1 = np.sqrt((gate_point_transformed[0][0][0] - ball_point_transformed[0][0][0]) ** 2 + (gate_point_transformed[0][0][1] - ball_point_transformed[0][0][1]) ** 2)
    distance_2 = np.sqrt((left_foot_transformed[0][0][0] - gate_post_transformed[0][0][0]) ** 2 + (left_foot_transformed[0][0][1] - gate_post_transformed[0][0][1]) ** 2)
    distance_3 = np.sqrt((ball_point_transformed[0][0][0] - referee_point_transformed[0][0][0]) ** 2 + (ball_point_transformed[0][0][1] - referee_point_transformed[0][0][1]) ** 2)
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