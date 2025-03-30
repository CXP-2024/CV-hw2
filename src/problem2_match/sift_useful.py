import cv2
import numpy as np
import matplotlib.pyplot as plt
from homography_manual import find_homography_manual, compute_transformed_coordinates, warp_perspective_manual
from gaussian import gaussian_filter



def create_gaussian_pyramid(image, num_octaves, scales_per_octave, sigma0=1.6):
    """
    Create a Gaussian pyramid with multiple octaves and scales.
    
    Args:
        image: Input grayscale image
        num_octaves: Number of octaves
        scales_per_octave: Number of scales per octave
        sigma0: Initial sigma
        
    Returns:
        gaussian_pyramid: List of octaves, each containing Gaussian blurred images
    """
    # # change to [0, 1] range
    # if image.max() > 1:
    #     image = image / 255.0

    k = 2 ** (1.0 / scales_per_octave)
    gaussian_pyramid = []
    
    # For each octave
    for octave in range(num_octaves):
        octave_images = []
        
        # Downsample image for this octave
        if octave == 0:
            octave_base = image.copy()
        else:
            octave_base = cv2.resize(gaussian_pyramid[-1][0], 
                                     (gaussian_pyramid[-1][0].shape[1] // 2, 
                                     gaussian_pyramid[-1][0].shape[0] // 2))
        
        # Create scales for this octave
        for scale in range(scales_per_octave + 3):
            sigma = sigma0 * (k ** scale)
            kernel_size = int(2 * np.ceil(2 * sigma) + 1)
            blurred = gaussian_filter(octave_base, kernel_size, sigma=sigma)
            octave_images.append(blurred)
            
        gaussian_pyramid.append(octave_images)
        
    
    # show the gaussian pyramid, one plot for each octave, total plot number is num_octaves
    '''
    for i in range(num_octaves):
        plt.figure(figsize=(100, 15))
        for j in range(scales_per_octave + 3):
            plt.subplot(1, scales_per_octave + 3, j + 1)
            plt.imshow(gaussian_pyramid[i][j], cmap='gray')
            # plt.axis('off')
        print("Octave: ", i, " Scale: ", scales_per_octave + 3, " Shape: ", gaussian_pyramid[i][j].shape)
        sigma_box = [sigma0 * (k ** scale) for scale in range(scales_per_octave + 2)]
        print("Sigma: ", sigma_box)
        #print("Gaussian kernel: size", kernel_size, "\n", generate_gaussian_kernel(kernel_size, sigma=sigma_box[0] * 5))
        plt.tight_layout()
        plt.show()
    '''

    return gaussian_pyramid

def create_dog_pyramid(gaussian_pyramid):
    """
    Create Difference-of-Gaussian (DoG) pyramid from Gaussian pyramid.
    
    Args:
        gaussian_pyramid: Gaussian pyramid (list of octaves)
        
    Returns:
        dog_pyramid: List of DoG octaves containing difference images
    """
    dog_pyramid = []
    initial_shape = gaussian_pyramid[0][0].shape
    print("Initial Shape: ", initial_shape)
    #print("Initial Image: ", gaussian_pyramid[0][0])
    #print("Second Image: ", gaussian_pyramid[0][1])
    
    for octave_images in gaussian_pyramid:
        # Verify valid input structure
        if len(octave_images) < 2:
            raise ValueError("Octave must contain at least 2 Gaussian images")
            
        dog_images = []
        # Generate (n-1) DoG images per octave and remember to resize the image to the same size
        for i in range(1, len(octave_images)):
            dog = octave_images[i] - octave_images[i-1]
            #if i == 1:
              #print("max: ", np.max(dog), "shape: ", dog.shape, "DoG: ", dog)
            #dog = cv2.resize(dog, (initial_shape[1], initial_shape[0]))
            # change to [0, 1] range
            dog = (dog - np.min(dog)) / (np.max(dog) - np.min(dog)) 
            #if i == 1:
              #print("max: ", np.max(dog), "shape: ", dog.shape, "DoG: ", dog)
            dog_images.append(dog)
        dog_pyramid.append(dog_images)
    
    # show the DoG pyramid, one plot for each octave, total plot number is num_octaves
    '''
    for i in range(len(dog_pyramid)):
        plt.figure(figsize=(100, 15))
        for j in range(len(dog_pyramid[i])):
            plt.subplot(1, len(dog_pyramid[i]), j + 1)
            plt.imshow(dog_pyramid[i][j], cmap='gray')
						# plt.axis('off')
        print("Octave: ", i, " Scale: ", len(dog_pyramid[i]), " Shape: ", dog_pyramid[i][j].shape)
        plt.tight_layout()
        plt.show()
    '''

    return dog_pyramid

def detect_keypoints2(dog_pyramid, contrast_threshold=0.5, edge_threshold=10):
    """
    Detect keypoints in the DoG pyramid with sub-pixel refinement.
    
    Args:
        dog_pyramid: DoG pyramid from create_dog_pyramid()
        contrast_threshold: Minimum contrast (normalized to [0,1])
        edge_threshold: Eigenvalue ratio threshold (usually 10)
        
    Returns:
        keypoints: List of keypoints as (octave, scale, y, x)
    """
    keypoints = []
    #contrast_threshold = contrast_threshold * 255  # Scale to pixel values
    
    total_keypoints_num = 0
    for octave_idx, dog_octave in enumerate(dog_pyramid):
        # Require at least 3 DoG images for scale-space extremum
        if len(dog_octave) < 3:
            continue
        
        octave_num = 0
        for scale_idx in range(1, len(dog_octave)-1):
            prev_dog = dog_octave[scale_idx-1]
            curr_dog = dog_octave[scale_idx]
            next_dog = dog_octave[scale_idx+1]
            
            # Iterate through interior pixels (excluding 1-pixel border)
            height, width = curr_dog.shape
            for i in range(1, height-1):
                for j in range(1, width-1):
                    # Current pixel value
                    val = curr_dog[i, j]

                    
                    # 3D neighborhood check (3x3x3 cube)
                    neighborhood = np.concatenate([
                        prev_dog[i-1:i+2, j-1:j+2].flatten(),
                        curr_dog[i-1:i+2, j-1:j+2].flatten(),
                        next_dog[i-1:i+2, j-1:j+2].flatten()
                    ])  
                    neighborhood = np.delete(neighborhood, 13)  # Remove the center element
                    is_max = val > np.max(neighborhood)
                    is_min = val < np.min(neighborhood)

                    if not (is_max or is_min):
                        continue

                    # Edge response check using Hessian matrix
                    # Second derivatives (central differences)
                    dx = (curr_dog[i, j+1] - curr_dog[i, j-1]) / 2.0
                    dy = (curr_dog[i+1, j] - curr_dog[i-1, j]) / 2.0
                    dxx = curr_dog[i, j+1] + curr_dog[i, j-1] - 2*val
                    dyy = curr_dog[i+1, j] + curr_dog[i-1, j] - 2*val
                    dxy = (curr_dog[i+1, j+1] - curr_dog[i+1, j-1] - 
                          curr_dog[i-1, j+1] + curr_dog[i-1, j-1]) / 4.0
                    
                     # Solve for offset
                    gradient = np.array([dx, dy])
                    hessian = np.array([[dxx, dxy], [dxy, dyy]])
                    
                    try:
                        offset = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
                    except np.linalg.LinAlgError:
                        continue
                        
                    if np.abs(offset[0]) > 0.5 or np.abs(offset[1]) > 0.5:
                        continue  # Reject unstable refinements

                    # 3. Contrast thresholding AFTER refinement
                    contrast = curr_dog[i, j] + 0.5 * np.dot(gradient, offset)
                    if np.abs(contrast) < contrast_threshold:
                        continue

                    # 4. Edge rejection
                    tr = dxx + dyy
                    det = dxx * dyy - dxy**2
                    if det <= 0 or tr**2 * edge_threshold >= (edge_threshold + 1)**2 * det:
                        continue

                    # Store refined coordinates
                    keypoints.append((
                        octave_idx,
                        scale_idx,
                        i + offset[1],  # y-coordinate
                        j + offset[0]   # x-coordinate
                    ))
                    octave_num = octave_num + 1
                    
        print("find keypoints in octave: ", octave_idx, " keypoints number: ", octave_num)
        total_keypoints_num = total_keypoints_num + octave_num
    print("total keypoints number: ", total_keypoints_num)
                
    if len(keypoints) < 100:
        print("Warning: Very few keypoints detected, less than 100")
        print("redo the keypoints detection...", " last contrast_threshold: ", contrast_threshold)
        keypoints = detect_keypoints2(dog_pyramid, contrast_threshold=0.8* contrast_threshold, 
                                      edge_threshold=10)
        
    if len(keypoints) > 3500 and contrast_threshold < 0.9:
        print("Warning: Too many keypoints detected, more than 3500")
        print("redo the keypoints detection...", "last contrast_threshold: ", contrast_threshold)
        keypoints = detect_keypoints2(dog_pyramid, contrast_threshold=1.05 * contrast_threshold, edge_threshold=10)
        
    last_detected_number = len(keypoints)
    return keypoints

def compute_orientations2(gaussian_pyramid, keypoints, num_bins=36, sigma0=1.4):
    """
    Compute dominant orientations for each keypoint.
    
    Args:
        gaussian_pyramid: Gaussian pyramid
        keypoints: List of keypoints (octave, scale, y, x)
        num_bins: Number of orientation bins
        
    Returns:
        oriented_keypoints: List of keypoints with orientation (octave, scale, y, x, orientation)
    """
    oriented_keypoints = []
    
    for keypoint in keypoints:
        octave_idx, scale_idx, refine_y, refine_x = keypoint
        img = gaussian_pyramid[octave_idx][scale_idx]
        scales_per_octave = len(gaussian_pyramid[0]) - 2
        x = int(refine_x)
        y = int(refine_y)
        
        # Create histogram of orientations
        histogram = np.zeros(num_bins)
        # Gaussian weighting sigma0=1.6
        k = 2 ** (1.0 / scales_per_octave)
        sigma = sigma0 * (k ** scale_idx)
        radius = int(3 * sigma)
        
        for i in range(-radius, radius + 1):
            yi = y + i
            if yi <= 0 or yi >= img.shape[0] - 1:
                continue
                
            for j in range(-radius, radius + 1):
                xi = x + j
                if xi <= 0 or xi >= img.shape[1] - 1:
                    continue
                
                # Compute gradient
                dx = img[yi, xi+1] - img[yi, xi-1]
                dy = img[yi+1, xi] - img[yi-1, xi]
                
                # Compute magnitude and orientation
                magnitude = np.sqrt(dx**2 + dy**2)
                orientation = np.arctan2(dy, dx) % (2 * np.pi)
                
                # Weight by magnitude and distance from center
                weight = magnitude * np.exp(-(i**2 + j**2) / (2 * sigma**2))
                
                # Add to histogram
                bin_idx = int(orientation / (2 * np.pi) * num_bins) % num_bins
                histogram[bin_idx] += weight
        
        # Smooth histogram
        histogram = np.roll(histogram, 1) + histogram + np.roll(histogram, -1)
        histogram = histogram / 3.0
        
        # Find peaks in histogram
        threshold = 0.8 * np.max(histogram)
        for bin_idx in range(num_bins):
            if (histogram[bin_idx] > histogram[(bin_idx-1) % num_bins] and
                histogram[bin_idx] > histogram[(bin_idx+1) % num_bins] and
                histogram[bin_idx] >= threshold):
                
                # Convert bin index to angle (in radians)
                angle = bin_idx * 2 * np.pi / num_bins
                oriented_keypoints.append((octave_idx, scale_idx, refine_y, refine_x, angle))
    
    return oriented_keypoints

def compute_descriptors2(gaussian_pyramid, oriented_keypoints, descriptor_size=4, num_bins=8, sigma0=1.4):
    """
    Compute SIFT descriptors for keypoints.
    
    Args:
        gaussian_pyramid: Gaussian pyramid
        oriented_keypoints: List of keypoints with orientation
        descriptor_size: Size of descriptor grid (e.g., 4 means 4x4 grid)
        num_bins: Number of orientation bins per grid cell
        
    Returns:
        keypoints: List of keypoint locations (x, y, scale, orientation)
        descriptors: List of descriptors
    """
    # This function do these things:
    # 1. Compute the window size * size , size is 4 * 3 *sigma as int
    # 2. Get the orientation of the keypoint from the oriented_keypoints
    # 3. slice the window into 4 * 4 cells, and each cell has 8 bins, so the descriptor is 4 * 4 * 8, each cell size is 3 * sigma as int
    # 4. For each cell, calculate the gradient magnitude and orientation, and then weight by magnitude and distance from center, according to the orientation, add to the corresponding bin, here we should add the orientation to histogram bins
		# 5. Threshold and normalize for illumination invariance
    # 6. Add the bins to the descriptor
    # 7. Add the descriptor to the descriptors_list, so is the keypoints_list
    keypoints_list = []
    descriptors_list = []
    
    for keypoint in oriented_keypoints:
        octave_idx, scale_idx, refine_y, refine_x, angle = keypoint
        img = gaussian_pyramid[octave_idx][scale_idx]
        scales_per_octave = len(gaussian_pyramid[0]) - 2
        x = int(refine_x)
        y = int(refine_y)
        
        scale = 2 ** octave_idx
        k = 2 ** (1.0 / scales_per_octave)
        sigma = sigma0 * (k  ** scale_idx)
        
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        cell_size = int(3 * sigma)
        half_width = (descriptor_size * cell_size) // 2
        
        descriptor = np.zeros((descriptor_size, descriptor_size, num_bins))
        
        for cell_i in range(descriptor_size):
            for cell_j in range(descriptor_size):
                for i in range(cell_size):
                    for j in range(cell_size):
                        # compute the rotated coordinates
                        u = (cell_i - descriptor_size/2) * cell_size + i
                        v = (cell_j - descriptor_size/2) * cell_size + j
                        x_rot = u * cos_angle - v * sin_angle
                        y_rot = u * sin_angle + v * cos_angle
                        sample_x = x + x_rot
                        sample_y = y + y_rot
                        
                        if sample_x < 1 or sample_x >= img.shape[1]-1 or sample_y < 1 or sample_y >= img.shape[0]-1:
                            continue
                        
                        dx = img[int(sample_y), int(sample_x)+1] - img[int(sample_y), int(sample_x)-1]
                        dy = img[int(sample_y)+1, int(sample_x)] - img[int(sample_y)-1, int(sample_x)]
                        magnitude = np.sqrt(dx**2 + dy**2)
                        orientation = (np.arctan2(dy, dx) - angle) % (2 * np.pi)
                        
                        bin_idx = int(orientation / (2 * np.pi) * num_bins) % num_bins
                        weight = magnitude * np.exp(-((i - cell_size//2)**2 + (j - cell_size//2)**2) / (2 * (0.5 * cell_size)**2))
                        descriptor[cell_i, cell_j, bin_idx] += weight
        
        flat_descriptor = descriptor.flatten()
        threshold = 0.2 * np.linalg.norm(flat_descriptor)
        flat_descriptor = np.minimum(flat_descriptor, threshold)
        norm = np.linalg.norm(flat_descriptor)
        if norm > 0:
            flat_descriptor /= norm
        
        keypoints_list.append((int(refine_x * scale), int(refine_y * scale), scale, angle))
        descriptors_list.append(flat_descriptor)
    
    return keypoints_list, np.array(descriptors_list)
   							         
def match_descriptors(desc1, desc2, ratio_threshold=0.75):
    """
    Match descriptors using ratio test.
    
    Args:
        desc1: First set of descriptors
        desc2: Second set of descriptors
        ratio_threshold: Ratio test threshold
        
    Returns:
        matches: List of matches (idx1, idx2)
    """
    if len(desc1) == 0 or len(desc2) == 0:
        print("No descriptors to match")
        return []
    matches = []
    
    for i, descriptor in enumerate(desc1):
        # Compute distances to all descriptors in desc2
        distances = []
        for descriptor2 in desc2:
            diff = descriptor - descriptor2
            distance = np.sqrt(np.sum(diff**2))
            distances.append(distance)
        
        # Find indices of two closest matches
        idx = np.argsort(distances)
        
        # Apply ratio test
        if distances[idx[0]] < ratio_threshold * distances[idx[1]]:
            matches.append((i, idx[0]))
    
    return matches

def ransac_match(keypoints1, keypoints2, matches, num_iterations=1000, inlier_threshold=10):
    """
    Match keypoints between two images using RANSAC.
    
    Args:
        keypoints1: List of keypoints from first image
        descriptors1: List of descriptors from first image
        keypoints2: List of keypoints from second image
        descriptors2: List of descriptors from second image
        matches: List of matches (idx1, idx2)
        num_iterations: Number of RANSAC iterations        
        inlier_threshold: Threshold for inliers    

    Returns:    
        best_inliers: List of best inlier matches
        best_homography: Best homography matrix
    """
    import random
    
    # Need at least 4 points to compute homography
    if len(matches) < 4:
        return [], None
    
    # Extract matched points
    src_pts = np.float32([keypoints1[match[0]][:2] for match in matches])
    dst_pts = np.float32([keypoints2[match[1]][:2] for match in matches])
    
    best_inliers = []
    best_homography = None
    
    for _ in range(num_iterations):
        # Randomly select 4 matches
        sample_indices = random.sample(range(len(matches)), 4)
        
        # Extract the points from these matches
        src_sample = np.float32([src_pts[i] for i in sample_indices])
        dst_sample = np.float32([dst_pts[i] for i in sample_indices])
        
        try:
            # Compute homography using the 4 points
            homography = find_homography_manual(src_sample, dst_sample)
            
            # Skip if homography couldn't be computed
            if homography is None:
                continue
                
            # Count inliers
            inliers = []
            
            for i, (src, dst) in enumerate(zip(src_pts, dst_pts)):
                # Apply homography to source point
                src_transformed = np.dot(homography, np.array([src[0], src[1], 1]))
                if src_transformed[2] == 0:  # Check for division by zero
                    continue
                    
                # Convert to (x, y) coordinates
                src_transformed = src_transformed[:2] / src_transformed[2]
                
                # Calculate distance
                dist = np.sqrt(np.sum((src_transformed - dst)**2))
                
                # Check if it's an inlier
                if dist < inlier_threshold:
                    inliers.append(matches[i])
            
            # Update best result if this has more inliers
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_homography = homography
                
        except Exception as e:
            # Skip this iteration if there's an error
            continue
    # if we have enough inliers, we can compute the homography again
    if len(best_inliers) >= 4:
        src_pts = np.float32([keypoints1[match[0]][:2] for match in best_inliers])
        dst_pts = np.float32([keypoints2[match[1]][:2] for match in best_inliers])
        best_homography = find_homography_manual(src_pts, dst_pts)
    return best_inliers, best_homography
        
def visualize_keypoints(image, keypoints):
	"""
	Visualize keypoints on an image.
	
	Args:
		image: Input image
		keypoints: List of keypoints (x, y, scale, orientation)
	"""
	plt.figure(figsize=(10, 8))
	plt.imshow(image, cmap='gray')
	
	# Create a color cycle for different keypoints
	colors = plt.cm.hsv(np.linspace(0, 1, len(keypoints)))
	
	for i, kp in enumerate(keypoints):
		x, y, scale, orientation = kp
		
		radius = scale * 3
		
		# Draw keypoint circle with unique color
		circle = plt.Circle((x, y), radius, fill=False, color=colors[i])
		plt.gca().add_patch(circle)
		
		# Draw orientation line with same color
		line_x = x + radius * np.cos(orientation)
		line_y = y + radius * np.sin(orientation)
		plt.plot([x, line_x], [y, line_y], color=colors[i])
	
	plt.axis('off')
	plt.title("keypoints num: " + str(len(keypoints))) # Use f-string formatting
	plt.tight_layout()
	plt.show()

def visualize_matches(img1, kp1, img2, kp2, matches):
	"""
	Visualize matches between two images.
	
	Args:
		img1: First image
		kp1: Keypoints in first image
		img2: Second image
		kp2: Keypoints in second image
		matches: List of matches (idx1, idx2)
	"""
	# Create a new image with both images side by side
	h1, w1 = img1.shape[:2]
	h2, w2 = img2.shape[:2]
	
	h = max(h1, h2)
	w = w1 + w2
	
	vis = np.zeros((h, w), dtype=np.uint8)
	vis[:h1, :w1] = img1 if len(img1.shape) == 2 else cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
	vis[:h2, w1:w1+w2] = img2 if len(img2.shape) == 2 else cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
	
	plt.figure(figsize=(12, 8))
	plt.imshow(vis, cmap='gray')
	
	# Create color map
	colors = plt.cm.hsv(np.linspace(0, 1, len(matches)))
	
	# Draw lines between matches
	for i, match in enumerate(matches):
		idx1, idx2 = match
		x1, y1 = kp1[idx1][0], kp1[idx1][1]
		x2, y2 = kp2[idx2][0] + w1, kp2[idx2][1]
		
		# Use a different color for each line
		plt.plot([x1, x2], [y1, y2], color=colors[i])
		
		# Draw small circles at each vertex
		plt.plot(x1, y1, 'o', color=colors[i], markersize=5)
		plt.plot(x2, y2, 'o', color=colors[i], markersize=5)
	
	plt.axis('off')
	plt.title("Matches: " + str(len(matches))) # Use f-string formatting
	plt.tight_layout()
	plt.show()

def sift(gray_image, num_octaves=4, scales_per_octave=4, contrast_threshold=0.6, edge_threshold=10):
    """
    SIFT feature detection and description pipeline.
    
    Args:
        image: Input image
        num_octaves: Number of octaves
        scales_per_octave: Number of scales per octave
        contrast_threshold: Threshold for low contrast keypoints
        edge_threshold: Threshold for edge response
        
    Returns:
        keypoints: List of keypoint locations
        descriptors: Array of descriptors
    """
    # If not a grayscale image, convert it
    if len(gray_image.shape) > 2:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_RGB2GRAY)
        
		# If not a float image, convert it
    if gray_image.dtype != np.float32:
        gray_image = gray_image.astype(np.float32)
    
    # Create Gaussian pyramid
    gaussian_pyr = create_gaussian_pyramid(gray_image, num_octaves, scales_per_octave)
    
    # Create DoG pyramid
    dog_pyr = create_dog_pyramid(gaussian_pyr)
    
    # Detect keypoints
    keypoints = detect_keypoints2(dog_pyr, contrast_threshold, edge_threshold)
    
    # Compute orientations
    oriented_keypoints = compute_orientations2(gaussian_pyr, keypoints)
    
    # Compute descriptors
    keypoints_list, descriptors = compute_descriptors2(gaussian_pyr, oriented_keypoints)
    
    return keypoints_list, descriptors

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

    def out(self, img, contrast_threshold=0.6, edge_threshold=10, num_octaves=4, scales_per_octave=4):
        """
        Implement Scale-Invariant Feature Transform(SIFT) algorithm
        :param img: float/int array, shape: (height, width, channel)
        :return sift_results (keypoints, descriptors)
        """
        # =========================================================================================================
        # TODO: Please fill this part with your code
        # But DO NOT change this interface
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift(gray_image, num_octaves, scales_per_octave, contrast_threshold, edge_threshold)
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

    def match(self, img1, img2):
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
        keypoints1, descriptors1 = sift(img1, num_octaves=4, scales_per_octave=4, contrast_threshold=0.6, edge_threshold=10)
        keypoints2, descriptors2 = sift(img2, num_octaves=4, scales_per_octave=4, contrast_threshold=0.6, edge_threshold=10)
        matches = match_descriptors(descriptors1, descriptors2, ratio_threshold=0.7)
        print("matchs number: ", len(matches))
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