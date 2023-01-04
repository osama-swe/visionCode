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

def detect_rocks(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([20,100,100])
    upper_yellow = np.array([40,255,255])
    mask = cv2.inRange(hsv,lower_yellow,upper_yellow)
    result = cv2.bitwise_and(img,img,mask = mask)
    binary_result = color_thresh(result,(0,0,0))
    return binary_result

# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dists = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dists, angles

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    # Apply a rotation
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))                         
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

# Define a function to perform a translation
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
    # obatin a perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # transform
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    return warped

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    dst = 5 
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[Rover.img.shape[1]/2 - dst, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst, Rover.img.shape[0] - 2*dst - bottom_offset], 
                  [Rover.img.shape[1]/2 - dst, Rover.img.shape[0] - 2*dst - bottom_offset]])

    # 2) Apply perspective transform to input image
    warped_navigable = perspect_transform(Rover.img, source, destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    threshed_navigable = color_thresh(warped_navigable)
    threshed_obstacle = 1-threshed_navigable
    threshed_rock = detect_rocks(warped_navigable)


    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    Rover.vision_image[:,:,0] = threshed_obstacle*255
    Rover.vision_image[:,:,1] = threshed_rock*255
    Rover.vision_image[:,:,2] = threshed_navigable*255

    #-----------------------------------------------------------------------------------
    threshed_navigable_crop = np.zeros_like(threshed_navigable)
    threshed_obstacle_crop = np.zeros_like(threshed_obstacle)
    x1 = np.int(threshed_navigable.shape[0]/2)
    x2 = np.int(threshed_navigable.shape[0])
    y1 = np.int(threshed_navigable.shape[1]/3)
    y2 = np.int(threshed_navigable.shape[1]*2/3)

    threshed_navigable_crop[x1:x2, y1:y2] = threshed_navigable[x1:x2, y1:y2]
    threshed_obstacle_crop[x1:x2, y1:y2]= threshed_obstacle[x1:x2, y1:y2]
    #-----------------------------------------------------------------------------------

    # 5) Convert map image pixel values to rover-centric coords
    xpix_nav, ypix_nav = rover_coords(threshed_navigable)
    xpix_obs, ypix_obs = rover_coords(threshed_obstacle)
    xpix_rock, ypix_rock = rover_coords(threshed_rock)


    xpix_nav_crop, ypix_nav_crop = rover_coords(threshed_navigable_crop)
    xpix_obs_crop, ypix_obs_crop = rover_coords(threshed_obstacle_crop)
 
    # 6) Convert rover-centric pixel values to world coordinates
    xpix_world_nav, ypix_world_nav = pix_to_world(xpix_nav_crop, 
                                                ypix_nav_crop, 
                                                Rover.pos[0], 
                                                Rover.pos[1], 
                                                Rover.yaw, 
                                                Rover.worldmap.shape[0], 
                                                scale = 10)
    xpix_world_obs, ypix_world_obs = pix_to_world(xpix_obs_crop, 
                                                ypix_obs_crop, 
                                                Rover.pos[0], 
                                                Rover.pos[1], 
                                                Rover.yaw, 
                                                Rover.worldmap.shape[0], 
                                                scale = 10)
    xpix_world_rock, ypix_world_rock = pix_to_world(xpix_rock, 
                                                ypix_rock, 
                                                Rover.pos[0], 
                                                Rover.pos[1], 
                                                Rover.yaw, 
                                                Rover.worldmap.shape[0], 
                                                scale = 10)

    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    if (np.float(np.abs(Rover.roll) % 360) <= 1) and (np.float(np.abs(Rover.pitch) % 360) <= 1):
        Rover.worldmap[ypix_world_obs, xpix_world_obs, 0] += 1
        Rover.worldmap[ypix_world_rock, xpix_world_rock, 1] += 1
        Rover.worldmap[ypix_world_nav, xpix_world_nav, 2] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    if (xpix_rock.any() or Rover.mode == 'goto_rock'):
        if (Rover.mode != 'goto_rock'):
            Rover.mode = 'goto_rock'
        if (xpix_rock.any()):
            Rover.nav_dists, Rover.nav_angles = to_polar_coords(xpix_rock,ypix_rock)
            Rover.see_rock_error = 0
        else:
            Rover.see_rock_error += 1
        if Rover.see_rock_error > 100:
            Rover.mode = 'stop'
    else:
        Rover.nav_dists, Rover.nav_angles = to_polar_coords(xpix_nav,ypix_nav)
        


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
        plt.imshow(warped_navigable)
        plt.title('Bird Eye View')
        plt.subplot(323)
        plt.imshow(threshed_navigable, cmap='gray')
        plt.title('Threshed image for terrain detection')
        plt.subplot(324)
        plt.plot(xpix_nav, ypix_nav, '.', color='blue')
        plt.ylim(-160, 160)
        plt.xlim(0, 160)
        arrow_length = 100
        mean_dir = np.mean(Rover.nav_angles)
        x_arrow = arrow_length * np.cos(mean_dir)
        y_arrow = arrow_length * np.sin(mean_dir)
        plt.arrow(0, 0, x_arrow, y_arrow, color='red', zorder=2, head_width=10, width=2)
        plt.title('Approximate direction where the Rover should move')
        plt.subplot(325)
        plt.imshow(threshed_obstacle, cmap='gray')
        plt.title('Threshed image for obstacles detection')
        plt.subplot(326)
        plt.plot(xpix_obs, ypix_obs, '.', color='green')
        plt.ylim(-160, 160)
        plt.xlim(0, 160)
        plt.title('Rover-centric obstacle pixels')
        plt.pause(1)

    generate_video = False
    if generate_video:
        plt.figure(2, figsize=(12,9))
        plt.clf()
        plt.subplot(321)
        fig1 = plt.imshow(Rover.img)
        plt.subplot(322)
        fig2 = plt.imshow(warped_navigable)
        plt.subplot(323)
        fig3 = plt.imshow(threshed_navigable, cmap='gray')
        plt.subplot(324)
        plt.plot(xpix_nav, ypix_nav, '.')
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


    return Rover
