# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 18:31:23 2022
Last modified: 17 June 2022
By: Fabian Kung
@author: user
"""

import os
#import time
from gpiozero import PWMLED
import serial
import time
import numpy as np
import pygame
from pygame import camera
from pygame import display

''' Declaration of Global Variables --- '''
TFLITE_MODEL_DIR = './TfLite_model'
PATH_TO_TFLITE_MODEL = os.path.join(TFLITE_MODEL_DIR,'model.tflite')

_SHOW_COLOR_IMAGE = False

# Original image size
#_imgwidth_ori = 640
#_imgheight_ori = 480
_imgwidth_ori = 320
_imgheight_ori = 240

# Set the width and height of the input image in pixels for tensorflow pipeline.
_imgwidth = 160
_imgheight = 120

# Set the region of interest start point and size.
# Note: The coordinate (0,0) starts at top left hand corner of the image frame.
_roi_startx = 30
_roi_starty = 71
_roi_width = 100
_roi_height = 37

_LED_LOW_BRIGHTNESS = 0.1  # Normal brightness of LED, 10%.
_LED_MEDIUM_BRIGHTNESS = 0.3
_LED_MAX_BRIGHTNESS = 0.5



pygame.init() # This initialize pygame, including the display as well.
camera.init()
mycam = camera.Camera(camera.list_cameras()[0],(_imgwidth_ori,_imgheight_ori),'HSV')
mycam.start()
screen = display.set_mode((_imgwidth_ori,_imgheight_ori))
display.set_caption("cam")
pwmLED1 = PWMLED(24)                    # Initialize GPIO24 to control LED.
pwmLED1.value = _LED_LOW_BRIGHTNESS     # Turn on LED at low brigthness.


rescale_level = int(_imgwidth_ori/_imgwidth) # Scale to reduce the image size.
nrow = int(_imgheight_ori/rescale_level)
ncol = int(_imgwidth_ori/rescale_level)
averaging_coeff = 1.0/rescale_level

Ave_row = np.zeros((nrow,_imgheight_ori),dtype=float)
Ave_col = np.zeros((_imgwidth_ori,ncol),dtype=float)
for row in range(nrow):
    for index in range(rescale_level):
        Ave_row[row,rescale_level*row+index] = averaging_coeff
   
for col in range(ncol):
    for index in range(rescale_level):
        Ave_col[rescale_level*col+index,col] = averaging_coeff

# Codes to calculate the coordinates for rectangles and other structures 
# that will be superimposed on the display screen as user feedback.
pointROIstart = (rescale_level*_roi_startx,rescale_level*_roi_starty)
pointROIsize = (rescale_level*_roi_width,rescale_level*_roi_height)
pgrectROI = pygame.Rect(pointROIstart,pointROIsize)

interval = np.floor(_roi_width/3)
interval2 = np.floor(2*_roi_width/3)
# Rectangle for label1 (object on left)
pointL1start = (rescale_level*(_roi_startx+4),rescale_level*(_roi_starty+4))
pointL1size = (rescale_level*(int(interval)-8),rescale_level*(_roi_height-8))
pgrectL1 = pygame.Rect(pointL1start,pointL1size)
# Rectangle for label2 (object on right)
pointL2start = (rescale_level*(_roi_startx+4+int(interval2)),rescale_level*(_roi_starty+4))
pointL2size = (rescale_level*(int(interval)-8),rescale_level*(_roi_height-8))
pgrectL2 = pygame.Rect(pointL2start,pointL2size)
# Rectangle for label3 (object in front)
pointL3start = (rescale_level*(_roi_startx+4+int(interval)),rescale_level*(_roi_starty+4))
pointL3size = (rescale_level*(int(interval)-8),rescale_level*(_roi_height-8))
pgrectL3 = pygame.Rect(pointL3start,pointL3size)
# Rectangle for label4 (object blocking front)
pointL4start = (rescale_level*(_roi_startx+4),rescale_level*(_roi_starty+4))
pointL4size = (rescale_level*(_roi_width-8),rescale_level*(_roi_height-8))
pgrectL4 = pygame.Rect(pointL4start,pointL4size)

# --- For PC ---
#import tensorflow as tf
#interpreter = tf.lite.Interpreter(PATH_TO_TFLITE_MODEL) # Load the TFLite model in TFLite Interpreter
# === For Raspberry Pi ---
import tflite_runtime.interpreter as tflite # Use tflite runtime instead of TensorFlow.
interpreter = tflite.Interpreter(PATH_TO_TFLITE_MODEL)

# There is only 1 signature defined in the model,
# so it will return it by default.
# If there are multiple signatures then we can pass the name.
my_signature = interpreter.get_signature_runner()

# Optional, show the format for input.
input_details = interpreter.get_input_details()
# input_details is a dictionary containing the details of the input
# to this neural network.
print(input_details[0])
print(input_details[0]['shape'])
# Now print the signature input and output names.
print(interpreter.get_signature_list())

        
    
is_running = True

while is_running:
    img = mycam.get_image()                             # Note: the return from get_image() is an object
                                                        # called Surface in pygame. This is a 2D array with the
                                                        # element being a 32-bits unsigned integer for the pixel
                                                        # where the 8-bits RGB (default format) components are 
                                                        # coded as follows:
                                                        # pixel_value = (Rx256x256) + (Bx256) + R
                                                        # The multiply by 256 corresponds to left shift 8-bits.
    imgnp = np.asarray(pygame.surfarray.array3d(img),dtype=np.uint32)   # Convert 2d surface into 3D array, with the last index
                                                        # points to the color component.
    #imgR = imgnp[:,:,0]                                 # Extract the R component.
    #imgG = imgnp[:,:,1]                                 # Extract the G component.
    imgI = imgnp[:,:,2]                                 # Extract the V component.
    
    imgIt = np.transpose(imgI)                          # Flip the image array to the correct orientation.
    imgIresize = np.matmul(imgIt,Ave_col)               # Perform image resizing using averaging method.
                                                        # To speed up the process, instead of using dual for-loop,
                                                        # we use numpy matrix multiplication method. Here we 
    imgIresize = np.matmul(Ave_row,imgIresize)          # multiply the image matrix on left and right hand side
                                                        # This performs averaging allow the row and column while
                                                        # reducing the width and height of the original image matrix.
    # Crop out region-of-interest (ROI)
    imggrayresizecrop = imgIresize[_roi_starty:_roi_starty+_roi_height,_roi_startx:_roi_startx+_roi_width] 
    # Normalize each pixel value to floating point, between 0.0 to +1.0
    # NOTE: This must follows the original mean and standard deviation 
    # values used in the TF model. Need to refer to the model pipeline.
    # In Tensorflow, the normalization is done by the detection_model.preprocess(image) 
    # method. In TensorFlow lite we have to do this explicitly. 
    imggrayresizecropnorm = imggrayresizecrop/256.0                 # Normalized to 32-bits floating points 
    # --- Method 1 using tf.convert_to_tensor to make a tensor from the numpy array ---
    #input_tensor = tf.convert_to_tensor(test, dtype=tf.float32)

    # --- Method 2 to prepare the input, only using numpy ---
    input_tensor = np.asarray(np.expand_dims(imggrayresizecropnorm,(0,-1)), dtype = np.float32)

    output = my_signature(conv2d_input = input_tensor)  # Perform inference on the input. The input and 
                                                    # output names can
                                                    # be obtained from interpreter.get_signature_list()

    output1 = np.squeeze(output['dense_1'])         # Remove 1 dimension from the output. The output 
                                                    # parameters are packed into a dictionary. With 
                                                    # the name 'dense_1' to access the output layer. 
    result = np.argmax(output1)    
    #print(result)
    
    imgnp[:,:,0] = imgI                             # Create a gray-scale image array by duplicating the luminance V
    imgnp[:,:,1] = imgI                             # values on channel 0 and channel 1 of the 3D image array.
    pygame.surfarray.blit_array(screen,imgnp)       # Copy 3D image array to display surface using block transfer.
                              
    # Draw the ROI border on the screen.
    pygame.draw.rect(screen,(0,0,255),pgrectROI,width=rescale_level) 
    # Draw rectangle for Label 1 to 4 in ROI    
    if result == 1:
        pwmLED1.value = _LED_MEDIUM_BRIGHTNESS
        pygame.draw.rect(screen,(255,255,0),pgrectL1,width=rescale_level)
    elif result == 2:
        pwmLED1.value = _LED_MEDIUM_BRIGHTNESS
        pygame.draw.rect(screen,(255,255,0),pgrectL2,width=rescale_level)
    elif result == 3:
        pwmLED1.value = _LED_MEDIUM_BRIGHTNESS
        pygame.draw.rect(screen,(255,255,0),pgrectL3,width=rescale_level)
    elif result == 4:
        pwmLED1.value = _LED_MAX_BRIGHTNESS
        pygame.draw.rect(screen,(255,255,0),pgrectL4,width=rescale_level)
    else:
        pwmLED1.value = _LED_LOW_BRIGHTNESS     # Turn on LED at low brigthness.
        
    display.update()                                    # This will create a window and display the image.
    #display.flip()
    for event in pygame.event.get():  # Just close the window and a QUIT even will be generated
        if event.type == pygame.QUIT:
            is_running = False
mycam.stop()
pygame.quit()
pwmLED1.off()               # Turn off LED to indicate that program is no longer executing.

