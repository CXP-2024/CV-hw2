import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, convolve
import cv2  # Only for basic image operations

def rgb2gray(image):
    """Convert RGB image to grayscale"""
    if len(image.shape) == 3:
        return np.dot(image[..., :3], [0.299, 0.587, 0.114])
    return image

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
            blurred = gaussian_filter(octave_base, sigma)
            octave_images.append(blurred)
            
        gaussian_pyramid.append(octave_images)
    
    # show the gaussian pyramid
    plt.figure(figsize=(10, 8))
    for i, octave_images in enumerate(gaussian_pyramid):
        plt.subplot(1, num_octaves, i + 1)
        plt.imshow(np.hstack(octave_images), cmap='gray')
        plt.axis('off')
        plt.title(f'Octave {i+1}')
    plt.tight_layout()
    plt.show()

    return gaussian_pyramid

def create_dog_pyramid(gaussian_pyramid):
    """
    Create Difference-of-Gaussian (DoG) pyramid from Gaussian pyramid.
    
    Args:
        gaussian_pyramid: Gaussian pyramid
        
    Returns:
        dog_pyramid: List of DoG images
    """
    dog_pyramid = []
    
    for octave_images in gaussian_pyramid:
        dog_images = []
        for i in range(1, len(octave_images)):
            # Compute difference between adjacent scales
            dog = octave_images[i] - octave_images[i-1]
            dog_images.append(dog)
        dog_pyramid.append(dog_images)
    
    return dog_pyramid

def detect_keypoints(dog_pyramid, contrast_threshold=0.03, edge_threshold=10):
    """
    Detect keypoints in the DoG pyramid.
    
    Args:
        dog_pyramid: DoG pyramid
        contrast_threshold: Threshold for low contrast keypoints
        edge_threshold: Threshold for edge response
        
    Returns:
        keypoints: List of keypoints (octave, scale, y, x)
    """
    keypoints = []
    
    for octave_idx, dog_octave in enumerate(dog_pyramid):
        for scale_idx in range(1, len(dog_octave) - 1):
            # Get the three adjacent DoG images
            prev_dog = dog_octave[scale_idx - 1]
            curr_dog = dog_octave[scale_idx]
            next_dog = dog_octave[scale_idx + 1]
            
            # Iterate through each pixel (excluding borders)
            for i in range(1, curr_dog.shape[0] - 1):
                for j in range(1, curr_dog.shape[1] - 1):
                    # Check if it's a local extremum
                    center_val = curr_dog[i, j]
                    
                    # Skip low contrast regions
                    if abs(center_val) < contrast_threshold:
                        continue
                    
                    # Create 3x3x3 cube around the point
                    cube = np.stack([
                        prev_dog[i-1:i+2, j-1:j+2],
                        curr_dog[i-1:i+2, j-1:j+2],
                        next_dog[i-1:i+2, j-1:j+2]
                    ])
                    
                    # Check if maximum or minimum in 3x3x3 neighborhood
                    if ((center_val == np.max(cube) or center_val == np.min(cube)) and
                          (center_val != cube[1, 1, 1])):
                        
                        # Compute Hessian matrix for edge response
                        dxx = curr_dog[i, j+1] + curr_dog[i, j-1] - 2 * center_val
                        dyy = curr_dog[i+1, j] + curr_dog[i-1, j] - 2 * center_val
                        dxy = ((curr_dog[i+1, j+1] - curr_dog[i+1, j-1]) - 
                              (curr_dog[i-1, j+1] - curr_dog[i-1, j-1])) / 4.0
                        
                        # Compute ratio of eigenvalues
                        tr = dxx + dyy
                        det = dxx * dyy - dxy * dxy
                        
                        # Skip edge-like features
                        if det <= 0 or (tr**2 / det) >= (edge_threshold + 1)**2 / edge_threshold:
                            continue
                        
                        keypoints.append((octave_idx, scale_idx, i, j))
    
    return keypoints

def compute_orientations(gaussian_pyramid, keypoints, num_bins=36):
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
        octave_idx, scale_idx, y, x = keypoint
        img = gaussian_pyramid[octave_idx][scale_idx]
        
        # Create histogram of orientations
        histogram = np.zeros(num_bins)
        sigma = 1.5  # Gaussian weighting sigma
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
                oriented_keypoints.append((octave_idx, scale_idx, y, x, angle))
    
    return oriented_keypoints

def compute_descriptors(gaussian_pyramid, oriented_keypoints, descriptor_size=4, num_bins=8):
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
    keypoints_list = []
    descriptors_list = []
    
    for keypoint in oriented_keypoints:
        octave_idx, scale_idx, y, x, angle = keypoint
        img = gaussian_pyramid[octave_idx][scale_idx]
        
        # Compute scale factor
        scale = 2 ** octave_idx
        
        # Precompute sine and cosine values for rotation
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # Determine descriptor radius in pixels
        radius = descriptor_size * 4  # Each descriptor cell is 4x4 pixels
        
        # Initialize descriptor array
        descriptor = np.zeros((descriptor_size, descriptor_size, num_bins))
        
        # Sample points around the keypoint
        for i in range(-radius, radius):
            for j in range(-radius, radius):
                # Rotate the sample point
                rot_i = i * cos_angle - j * sin_angle
                rot_j = i * sin_angle + j * cos_angle
                
                # Determine which descriptor cell this point falls in
                cell_i = int((rot_i + radius) / (2 * radius) * descriptor_size - 0.5)
                cell_j = int((rot_j + radius) / (2 * radius) * descriptor_size - 0.5)
                
                if 0 <= cell_i < descriptor_size and 0 <= cell_j < descriptor_size:
                    # Get the coordinates in the image
                    sample_y = y + i
                    sample_x = x + j
                    
                    if 0 <= sample_y < img.shape[0] - 1 and 0 <= sample_x < img.shape[1] - 1:
                        # Compute gradient
                        dx = img[sample_y, sample_x + 1] - img[sample_y, sample_x - 1]
                        dy = img[sample_y + 1, sample_x] - img[sample_y - 1, sample_x]
                        
                        # Rotate gradient relative to keypoint orientation
                        rot_dx = dx * cos_angle + dy * sin_angle
                        rot_dy = -dx * sin_angle + dy * cos_angle
                        
                        # Compute magnitude and orientation
                        magnitude = np.sqrt(rot_dx**2 + rot_dy**2)
                        orientation = np.arctan2(rot_dy, rot_dx) % (2 * np.pi)
                        
                        # Calculate orientation bin
                        bin_idx = int(orientation / (2 * np.pi) * num_bins) % num_bins
                        
                        # Weight by magnitude and distance from center
                        weight = magnitude * np.exp(-(i**2 + j**2) / (2 * (0.5 * radius)**2))
                        
                        # Add to descriptor
                        descriptor[cell_i, cell_j, bin_idx] += weight
        
        # Flatten and normalize descriptor
        flat_descriptor = descriptor.flatten()
        
        # Threshold and normalize for illumination invariance
        threshold = 0.2 * np.linalg.norm(flat_descriptor)
        flat_descriptor = np.minimum(flat_descriptor, threshold)
        norm = np.linalg.norm(flat_descriptor)
        
        if norm > 0:
            flat_descriptor /= norm
        
        # Add to result
        keypoints_list.append((x * scale, y * scale, scale, angle))
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
    matches = []
    
    for i, descriptor in enumerate(desc1):
        # Compute distances to all descriptors in desc2
        distances = np.sqrt(np.sum((desc2 - descriptor)**2, axis=1))
        
        # Find indices of two closest matches
        idx = np.argsort(distances)
        
        # Apply ratio test
        if distances[idx[0]] < ratio_threshold * distances[idx[1]]:
            matches.append((i, idx[0]))
    
    return matches

def visualize_keypoints(image, keypoints):
    """
    Visualize keypoints on an image.
    
    Args:
        image: Input image
        keypoints: List of keypoints (x, y, scale, orientation)
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='gray')
    
    for kp in keypoints:
        x, y, scale, orientation = kp
        
        radius = scale * 3
        
        # Draw keypoint circle
        circle = plt.Circle((x, y), radius, fill=False, color='r')
        plt.gca().add_patch(circle)
        
        # Draw orientation line
        line_x = x + radius * np.cos(orientation)
        line_y = y + radius * np.sin(orientation)
        plt.plot([x, line_x], [y, line_y], color='r')
    
    plt.axis('off')
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
    vis[:h1, :w1] = img1 if len(img1.shape) == 2 else rgb2gray(img1)
    vis[:h2, w1:w1+w2] = img2 if len(img2.shape) == 2 else rgb2gray(img2)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(vis, cmap='gray')
    
    # Draw lines between matches
    for match in matches:
        idx1, idx2 = match
        x1, y1 = kp1[idx1][0], kp1[idx1][1]
        x2, y2 = kp2[idx2][0] + w1, kp2[idx2][1]
        
        plt.plot([x1, x2], [y1, y2], 'c-')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def sift(gray_image, num_octaves=4, scales_per_octave=3, contrast_threshold=0.03, edge_threshold=10):
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
    
    # Create Gaussian pyramid
    gaussian_pyr = create_gaussian_pyramid(gray_image, num_octaves, scales_per_octave)
    
    # Create DoG pyramid
    dog_pyr = create_dog_pyramid(gaussian_pyr)
    
    # Detect keypoints
    keypoints = detect_keypoints(dog_pyr, contrast_threshold, edge_threshold)
    
    # Compute orientations
    oriented_keypoints = compute_orientations(gaussian_pyr, keypoints)
    
    # Compute descriptors
    keypoints_list, descriptors = compute_descriptors(gaussian_pyr, oriented_keypoints)
    
    return keypoints_list, descriptors
