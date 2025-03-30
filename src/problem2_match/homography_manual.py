import numpy as np

def find_homography_manual(src_points, dst_points):
    """
    Calculate the homography matrix from src_points to dst_points
    
    Args:
        src_points: Array of source points (N, 2)
        dst_points: Array of destination points (N, 2)
    
    Returns:
        homography: 3x3 homography matrix
    """
    ##############################################################################
		# The homography matrix h is defined as:
    #### u = h[0,0] * x + h[0,1] * y + h[0,2] / h[2, 0] * x + h[2,1] * y + h[2,2] #####
    #### v = h[1,0] * x + h[1,1] * y + h[1,2] / h[2, 0] * x + h[2,1] * y + h[2,2] #####
    #### where (x, y) are the coordinates of the source point and (u, v) are the coordinates of the destination point. (x, y, 1) to (uz, vz, z) where z = h[2,0] * x + h[2,1] * y + h[2,2]
    if len(src_points) != len(dst_points) or len(src_points) < 4:
        raise ValueError("At least 4 point correspondences are required")
    
    # Construct the coefficient matrix A
    A = []
    for i in range(len(src_points)):
        x, y = src_points[i][0], src_points[i][1]
        u, v = dst_points[i][0], dst_points[i][1]
        
        # Each point gives us two equations
        A.append([x, y, 1, 0, 0, 0, -x*u, -y*u, -u])
        A.append([0, 0, 0, x, y, 1, -x*v, -y*v, -v])
    
    A = np.array(A)
    
    # Solve the system using SVD
    # A is a 2N x 9 matrix, Ah = 0, h is homography array
    # we want to find the vector h that minimizes ||Ah||
    # when using SVD, we can find the solution by taking the last column of V in A = UDV^T
    # where U is an orthogonal matrix, D is a diagonal matrix, and V is an orthogonal matrix.
    # Then , the objective function ||Ah|| = ||UDV^Th|| = ||DV^Th|| = ||D V^T h|| = ||D h' ||, where h' = V^T h
		# Since D's singular values are sorted in descending order, when h' = (0, 0, 0, 0, 0, 0, 0, 0, 1), we obtain the minimum value of ||D^T h' || = sigma9, 
    # The solution: h = Vh' = V[:, -1], where V is the right singular vector matrix of A.
    # if sigma9 = 0, then we can find h such that ||Ah|| = 0.
    _, _, Vt = np.linalg.svd(A)
    
    # The solution is the last column of V
    h = Vt[-1, :].reshape(3, 3)
    
    # Normalize the homography matrix so that h[2,2] = 1
    h = h / h[2, 2]
    
    return h

# Example usage:
# homography, _ = find_homography_manual(src_sample, dst_sample)

def transform_single_point(homography, point):
    """
		Transform a single point using the homography matrix.
		
		Args:
						homography: 3x3 homography matrix
						point: Point to be transformed (x, y)
		
		Returns:
						transformed_point: Transformed point (x', y')
		"""
		# Convert point to homogeneous coordinates
    point_homogeneous = np.array([point[0], point[1], 1])
		
		# Apply the homography transformation
    transformed_point_homogeneous = np.dot(homography, point_homogeneous)
		
		# Convert back to Cartesian coordinates
    transformed_point = transformed_point_homogeneous[:2] / transformed_point_homogeneous[2]
		
    return transformed_point.astype(np.float32)  # Changed from int8 to float32

def compute_transformed_coordinates(homography, points):
    """
    Compute the transformed coordinates of points using the homography matrix.
    
    Args:
            homography: 3x3 homography matrix
            points: Array of points (N, 2)
    
    Returns:
            transformed_points: Transformed points (N, 2)
    """
    # Convert points to homogeneous coordinates
    num_points = points.shape[0]
    points_homogeneous = np.hstack((points, np.ones((num_points, 1))))
    
    # Apply the homography transformation
    transformed_points_homogeneous = np.dot(homography, points_homogeneous.T).T
    
    # Convert back to Cartesian coordinates
    for i in range(num_points):
        transformed_points_homogeneous[i] /= transformed_points_homogeneous[i, 2]
    transformed_points = transformed_points_homogeneous[:, :2].astype(np.float32)  # Changed from int8 to float32
    # Now the form is [[x, y], [x, y], ...]

    return transformed_points


import tqdm
def warp_perspective_manual(img, H, output_size):
    """
    Apply perspective transformation to an image
    
    Args:
        img: Input image (height, width, channels)
        H: 3x3 homography matrix
        output_size: Tuple of (width, height) for output image
        
    Returns:
        Transformed image
    """
    # Create output image (black background)
    width, height = output_size
    output = np.zeros((height, width, img.shape[2]), dtype=np.uint8)
    
    # Calculate inverse homography (for backward mapping)
    H_inv = np.linalg.inv(H)
    
    # For each pixel in the output image
    for y_out in tqdm.tqdm(range(height)):
        for x_out in range(width):
            # Apply inverse homography to get source coordinates
            p_out = np.array([x_out, y_out, 1])
            p_in = np.dot(H_inv, p_out)
            
            # Convert to Cartesian coordinates
            if p_in[2] != 0:  # Check for division by zero
                x_in = p_in[0] / p_in[2]
                y_in = p_in[1] / p_in[2]
                
                # Check if the point is within the source image boundaries
                if 0 <= x_in < img.shape[1] and 0 <= y_in < img.shape[0]:
                    # Safe rounding to ensure indices are within bounds
                    x_in_int = min(int(round(x_in)), img.shape[1]-1)
                    y_in_int = min(int(round(y_in)), img.shape[0]-1)
                    output[y_out, x_out] = img[y_in_int, x_in_int]
    
    return output