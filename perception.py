import numpy as np
import cv2
import matplotlib.pyplot as plt

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

def detect_rocks(img, rgb_thresh=(110, 110, 60)):
    color_select = np.zeros_like(img[:,:,0])
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    color_select[above_thresh] = 1
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    mask = cv2.warpPerspective(np.ones_like(img[:,:,0]), M, (img.shape[1], img.shape[0]))
    
    return warped, mask


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    dst = 5
    bottom_offset = 6
    source = np.float32([[14, 140],
                     [300, 140],
                     [200, 95],
                     [120, 95]])

    destination = np.float32([[Rover.img.shape[1] / 2 - dst, Rover.img.shape[0] - bottom_offset],
                            [Rover.img.shape[1] / 2 + dst, Rover.img.shape[0] - bottom_offset],
                            [Rover.img.shape[1] / 2 + dst, Rover.img.shape[0] - 2*dst - bottom_offset],
                            [Rover.img.shape[1] / 2 - dst, Rover.img.shape[0] - 2*dst - bottom_offset]])

    # 2) Apply perspective transform
    warped, mask = perspect_transform(Rover.img, source, destination)
    
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    threshed = color_thresh(warped, rgb_thresh=(160, 160, 160))
    tmp_threshed = color_thresh(warped, rgb_thresh=(120, 120, 120))
    obstacles_thresh = (np.ones_like(threshed)-tmp_threshed)*mask
    rocks_thresh = detect_rocks(warped)

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    Rover.vision_image[:, :, 0] = obstacles_thresh*255
    Rover.vision_image[:, :, 1] = rocks_thresh*255
    Rover.vision_image[:, :, 2] = threshed*255

    # 5) Convert map image pixel values to rover-centric coords
    xpix, ypix = rover_coords(threshed)
    xobstacles, yobstacles = rover_coords(obstacles_thresh)
    xrocks, yrocks = rover_coords(rocks_thresh)

    # 6) Convert rover-centric pixel values to world coordinates
    x_pix_world, y_pix_world = pix_to_world(xpix,
                                            ypix,
                                            Rover.pos[0],
                                            Rover.pos[1],
                                            Rover.yaw,
                                            Rover.worldmap.shape[0],
                                            2*dst)
    obstacle_x_world, obstacle_y_world = pix_to_world(xobstacles,
                                                        yobstacles,
                                                        Rover.pos[0],
                                                        Rover.pos[1],
                                                        Rover.yaw,
                                                        Rover.worldmap.shape[0],
                                                        2*dst)
    rock_x_world, rock_y_world = pix_to_world(xrocks,
                                                yrocks,
                                                Rover.pos[0],
                                                Rover.pos[1],
                                                Rover.yaw,
                                                Rover.worldmap.shape[0],
                                                2*dst)

    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
    Rover.worldmap[rock_y_world, rock_x_world, 1] += 10
    Rover.worldmap[y_pix_world, x_pix_world, 2] += 10

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
    Rover.nav_dists, Rover.nav_angles = to_polar_coords(xpix, ypix)

    # To activate debugging mode, set the debugging_mode flag to True
    # To deactivate debugging mode, set the debugging_mode flag to False
    # Note that you can not use manual mode during autonomous mode if the debugging mode is activated
    # To be able to use the manual mode during autonomous mode, deactivate debugging mode
    # To exit the code while in debugging mode, type ctrl+c two or three times in the terminal
    # or close the terminal window
    debugging_mode = False
    if debugging_mode:
        plt.figure(1, figsize=(10,12))
        plt.clf()
        plt.subplot(321)
        plt.imshow(Rover.img)
        plt.title('Rover image')
        plt.subplot(322)
        plt.imshow(warped)
        plt.title('Bird Eye View')
        plt.subplot(323)
        plt.imshow(threshed, cmap='gray')
        plt.title('Threshed image for terrain detection')
        plt.subplot(324)
        plt.plot(xpix, ypix, '.', color='blue')
        plt.ylim(-160, 160)
        plt.xlim(0, 160)
        arrow_length = 100
        mean_dir = np.mean(Rover.nav_angles)
        x_arrow = arrow_length * np.cos(mean_dir)
        y_arrow = arrow_length * np.sin(mean_dir)
        plt.arrow(0, 0, x_arrow, y_arrow, color='red', zorder=2, head_width=10, width=2)
        plt.title('Approximate direction where the Rover should move')
        plt.subplot(325)
        plt.imshow(obstacles_thresh, cmap='gray')
        plt.title('Threshed image for obstacles detection')
        plt.subplot(326)
        plt.plot(xobstacles, yobstacles, '.', color='green')
        plt.ylim(-160, 160)
        plt.xlim(0, 160)
        plt.title('Rover-centric obstacle pixels')
        plt.pause(1)

    generate_video = True
    if generate_video:
        plt.figure(2, figsize=(12,9))
        plt.clf()
        plt.subplot(321)
        fig1 = plt.imshow(Rover.img)
        plt.subplot(322)
        fig2 = plt.imshow(warped)
        plt.subplot(323)
        fig3 = plt.imshow(threshed, cmap='gray')
        plt.subplot(324)
        plt.plot(xpix, ypix, '.')
        plt.ylim(-160, 160)
        plt.xlim(0, 160)
        arrow_length = 100
        mean_dir = np.mean(Rover.nav_angles)
        x_arrow = arrow_length * np.cos(mean_dir)
        y_arrow = arrow_length * np.sin(mean_dir)
        fig4 = plt.arrow(0, 0, x_arrow, y_arrow, color='red', zorder=2, head_width=10, width=2)
        fig1.axes.get_xaxis().set_visible(False)
        fig1.axes.get_yaxis().set_visible(False)
        fig2.axes.get_xaxis().set_visible(False)
        fig2.axes.get_yaxis().set_visible(False)
        fig3.axes.get_xaxis().set_visible(False)
        fig3.axes.get_yaxis().set_visible(False)
        fig4.axes.get_xaxis().set_visible(False)
        fig4.axes.get_yaxis().set_visible(False)
        plt.savefig('./output/pipeline_{}.jpg'.format(Rover.counter), bbox_inches='tight', pad_inches = 0)
        Rover.counter += 1
        plt.pause(1)

    # if rocks_thresh.any():
    #     Rover.nav_dists, Rover.nav_angles = to_polar_coords(xrocks, yrocks)

    return Rover