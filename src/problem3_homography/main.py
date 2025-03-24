import cv2 


# =========================================================================================================
# TODO: you can add other functions here or files in this directory
# =========================================================================================================

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
    distance_1 = 0.
    distance_2 = 0.
    distance_3 = 0.
    pass
    # =========================================================================================================

    return distance_1, distance_2, distance_3


if __name__ == '__main__':
    img_path = 'football.png'

    img = cv2.imread(img_path)

    # =========================================================================================================
    # for testing this demo:
    distance_1, distance_2, distance_3 = calculate_distance(img)
    # =========================================================================================================
