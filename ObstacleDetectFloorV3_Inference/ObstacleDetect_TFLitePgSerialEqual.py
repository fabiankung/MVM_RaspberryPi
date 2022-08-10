# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 18:31:23 2022
Last modified: 18 June 2022
By: Fabian Kung
@author: user
1. Camera interface using Pygame library.
2. Performs obstacle detection inference using CNN5Class model with gray scale mean equalization.
   NOTE: Make sure the correct TFLite model is loaded.
3. Pre-process input image - gray scale mean equalization.
4. Serial communication enabled using thread. Report CNN output via serial port.
"""

import os
import time
from gpiozero import PWMLED
import serial
import threading
import numpy as np
import pygame
from pygame import camera
from pygame import display

''' Declaration of Global Variables --- '''
TFLITE_MODEL_DIR = './TfLite_model'                                  # Path to exported TensorFlow Lite model folder.
PATH_TO_TFLITE_MODEL = os.path.join(TFLITE_MODEL_DIR,'model.tflite') # NOTE: If run from command line,
                                                                     # we may need to type the full path and
                                                                     # not the relative path.

# Original image size
#_IMGWIDTH_ORI = 640
#_IMGHEIGHT_ORI = 480
_IMGWIDTH_ORI = 320
_IMGHEIGHT_ORI = 240

# Set the actual width and height of the input image in pixels for tensorflow pipeline.
_IMGWIDTH = 160
_IMGHEIGHT = 120

# Set the region of interest start point and size.
# Note: The coordinate (0,0) starts at top left hand corner of the image frame.
_ROI_STARTX = 30
_ROI_STARTY = 71
_ROI_WIDTH = 100
_ROI_HEIGHT = 37

_LED_LOW_BRIGHTNESS = 0.1  # Normal brightness of LED, 10%.
_LED_MEDIUM_BRIGHTNESS = 0.5
_LED_MAX_BRIGHTNESS = 0.9

"""
 --- Serial Port (UART) Communication Class ---
This class uses threading to send and receive a string of bytes via the serial port. It handles
two-way communication between this computer and the Robot Controller, using the protocol as
defined in the Robot Controller firmware.

Usage:
Assume we instantiate this class as and start the class as follows:
  HeadRC_Interface = SerialCommThread("RC Comm Handler")      # Robot Controller interface thread.
  HeadRC_Interface.start()

Suppose we want to send a string of characters "M0+00\n" via the serial port, this can be done as follows:
blnTXSuccess = HeadRC_Interface.sendtoRC(bytes([77,48,43,48,48,10]))

The return variable blnTXSuccess is true is the transmission is successful, false otherwise.

When no data string is being transmitted via serial port, this class will periodically poll the
Robot Controller sensors status and other states, so that this computer knows the state of the
external robot platform. 
"""
class SerialCommThread(threading.Thread):

    # Declare class level variables.
    nIRSensor0_mm = 255
    nIRSensor1_mm = 255
    nIRSensor2_mm = 255
    
    ReadData = []                       # Class level variable.
    IRsenindex = 0
    
    def __init__(self, name):           # Overload constructor.
        threading.Thread.__init__(self) # Run original constructor. 
        self.name = name
        self.stopped = False
        self.TXcount = 0
        self.RC_TX_Busy = False
        self.utf8TXcommandstring = ""
    
    def run(self):                      # Overload run() method.
        while self.stopped == False:
            if self.RC_TX_Busy == False: # If no data string to send to Robot Controller (RC), poll RC status.
                
                self.TXcount = sport.write(bytes([71,68 ,48+SerialCommThread.IRsenindex, 48,48,10])) #"GD000\n" Get robot controller
                #self.TXcount = sport.write(bytes([71,68 ,48, 48,48,10])) #"GD000\n" Get robot controller
                                                                    # IR distance sensor output.
                                                                    # The current sensor index is stored in IRsenindex.
                ReadData = sport.read(5)                            # Read 5 bytes from RC.
                sport.reset_input_buffer()                          # Flush serial port

                # Check return data string format from Robot Controller and convert the data string to
                # binary value, and store in the appropraite global variable.
                if len(ReadData) > 0:                               # Make sure there is data before processing input string.
                    if ReadData[0] == 68:                           # Check if 1st character is 'D'
                        hundred = ReadData[1] - 48                  # Convert ASCII character to integer.
                        ten = ReadData[2] - 48
                        digit = ReadData[3] - 48
                        if SerialCommThread.IRsenindex == 0:
                            SerialCommThread.nIRSensor0_mm = digit + (ten*10) + (hundred*100)
                            #print("Sensor 0 = %d" % SerialCommThread.nIRSensor0_mm)
                        elif SerialCommThread.IRsenindex == 1:
                            SerialCommThread.nIRSensor1_mm = digit + (ten*10) + (hundred*100)
                            #print("Sensor 1 = %d" % SerialCommThread.nIRSensor1_mm)
                        else:
                            SerialCommThread.nIRSensor2_mm = digit + (ten*10) + (hundred*100)
                            #print("Sensor 2 = %d" % SerialCommThread.nIRSensor2_mm)

                SerialCommThread.IRsenindex = SerialCommThread.IRsenindex + 1   # Increment the IR sensor index. 
                if SerialCommThread.IRsenindex > 2:                             # Rest to 0 if index = 0.  
                    SerialCommThread.IRsenindex = 0
                
                
            else:                                                # If there is data to send to RC.
                #print(self.utf8TXcommandstring)
                self.TXcount = sport.write(self.utf8TXcommandstring)    
                ReadData = sport.read(2)                         # Read 2 bytes from RC, "OK".
                sport.reset_input_buffer()                       # Flush serial port 
                self.RC_TX_Busy = False                          # Clear busy flag.
                     
            time.sleep(0.02)                                     # 20 msec delay. So this thread will
                                                                 # Check for data string to be send to the RC
                                                                 # or get updates on the RC status 50 times
                                                                 # a second, which is good (earlier version
                                                                 # is 10 times/second).
        
    def stop(self):
        # Indicate that the thread should be stopped.
        self.stopped = True
        sport.close()                                           # Note: 8 May 2021. I noticed that if we close the
                                                                # COM port inside the thread, it may raise exception
                                                                # during the execution as other threads may still be
                                                                # using the COM port. Thus other methods in this thread
                                                                # needs to check this flag too.
        print("Port %s is closed" % sport.port)
        
    def sendtoRC(self, string):
        if self.RC_TX_Busy == False and self.stopped == False:
            self.RC_TX_Busy = True                              # Tell the run() thread there is data to send to RC.
            self.utf8TXcommandstring = string
            return True
        else:
            return False

#time.sleep(10) # Sleep for 5 seconds.

# 1). Setup serial port 
sport = serial.Serial()
sport.baudrate = 115200
sport.port = '/dev/ttyS0' # Tested ok for raspberry pi 3B and 3A+
# sport.port = 'COM33' # Tested ok for laptop with USB-to-Serial converter
sport.stopbits = serial.STOPBITS_ONE
sport.bytesize = serial.EIGHTBITS
sport.timeout = 0.01  # Read timeout value, 0.01 seconds or 10 msec.
                      # The external controller typically responds within 3 msec
sport.write_timeout = 0.02    # write timeout value, 0.02 second.
sport.xonxoff = False  # No hardware and software hand-shaking.
sport.dsrdtr = False
sport.rtscts = False
sport.open()
if sport.is_open == True:
    print("Port %s is opened" % sport.port)
else:
    print("Port %s not opened" % sport.port)
   
# 2). Setup pygame, camera and indicator LED 
pygame.init() # This initialize pygame, including the display as well.
camera.init()
mycam = camera.Camera(camera.list_cameras()[0],(_IMGWIDTH_ORI,_IMGHEIGHT_ORI),'HSV') # Return image in HSV colar space.
mycam.start()
screen = display.set_mode((_IMGWIDTH_ORI,_IMGHEIGHT_ORI))
display.set_caption("Camera View")
pwmLED1 = PWMLED(24)                    # Initialize GPIO24 to control LED.
pwmLED1.value = _LED_LOW_BRIGHTNESS     # Turn on LED at low brigthness.

# 3). Calculate the scale and other parameters of the image for analysis 
rescale_level = int(_IMGWIDTH_ORI/_IMGWIDTH) # Scale to reduce the image size for actual processing.
nrow = int(_IMGHEIGHT_ORI/rescale_level)     # Number of row
ncol = int(_IMGWIDTH_ORI/rescale_level)      # Number of column
averaging_coeff = 1.0/rescale_level          # Coefficient for matrix averaging operation. 

Ave_row = np.zeros((nrow,_IMGHEIGHT_ORI),dtype=float)
Ave_col = np.zeros((_IMGWIDTH_ORI,ncol),dtype=float)
for row in range(nrow):
    for index in range(rescale_level):
        Ave_row[row,rescale_level*row+index] = averaging_coeff
   
for col in range(ncol):
    for index in range(rescale_level):
        Ave_col[rescale_level*col+index,col] = averaging_coeff

# Codes to calculate the coordinates for rectangles and other structures 
# that will be superimposed on the display screen as user feedback.
pointROIstart = (rescale_level*_ROI_STARTX,rescale_level*_ROI_STARTY)
pointROIsize = (rescale_level*_ROI_WIDTH,rescale_level*_ROI_HEIGHT)
pgrectROI = pygame.Rect(pointROIstart,pointROIsize)

interval = np.floor(_ROI_WIDTH/3)
interval2 = np.floor(2*_ROI_WIDTH/3)
# Rectangle for label1 (object on left)
pointL1start = (rescale_level*(_ROI_STARTX+4),rescale_level*(_ROI_STARTY+4))
pointL1size = (rescale_level*(int(interval)-8),rescale_level*(_ROI_HEIGHT-8))
pgrectL1 = pygame.Rect(pointL1start,pointL1size)
# Rectangle for label2 (object on right)
pointL2start = (rescale_level*(_ROI_STARTX+4+int(interval2)),rescale_level*(_ROI_STARTY+4))
pointL2size = (rescale_level*(int(interval)-8),rescale_level*(_ROI_HEIGHT-8))
pgrectL2 = pygame.Rect(pointL2start,pointL2size)
# Rectangle for label3 (object in front)
pointL3start = (rescale_level*(_ROI_STARTX+4+int(interval)),rescale_level*(_ROI_STARTY+4))
pointL3size = (rescale_level*(int(interval)-8),rescale_level*(_ROI_HEIGHT-8))
pgrectL3 = pygame.Rect(pointL3start,pointL3size)
# Rectangle for label4 (object blocking front)
pointL4start = (rescale_level*(_ROI_STARTX+4),rescale_level*(_ROI_STARTY+4))
pointL4size = (rescale_level*(_ROI_WIDTH-8),rescale_level*(_ROI_HEIGHT-8))
pgrectL4 = pygame.Rect(pointL4start,pointL4size)

# 4). Load TensorFlow Lite model and setup 
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

# 5). Start all threads.
HeadRC_Interface = SerialCommThread("RC Comm Handler")      # Robot Controller interface thread.
HeadRC_Interface.start()
     
_totalpixel = (_IMGWIDTH-1)*(_IMGHEIGHT-1) 
   
# --- Main Loop ---    
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
    imgI = imgnp[:,:,2]                                 # Extract the V component.
    
    imgIt = np.transpose(imgI)                          # Flip the image array to the correct orientation.
    imgIresize = np.matmul(imgIt,Ave_col)               # Perform image resizing using averaging method.
                                                        # To speed up the process, instead of using dual for-loop,
                                                        # we use numpy matrix multiplication method. Here we 
    imgIresize = np.matmul(Ave_row,imgIresize)          # multiply the image matrix on left and right hand side
                                                        # This performs averaging allow the row and column while
                                                        # reducing the width and height of the original image matrix.
                                                        
    #--- Perform Equalization of Gray Level Mean ---    
    histo = np.zeros(256,dtype=int)     # Array for gray level histogram bin.    
            
    for i in range(_IMGHEIGHT-1):      # Update histogram bins.
        for j in range(_IMGWIDTH-1):
            pixel = int(imgIresize[i,j]) # Note that the content in the 2D array from pygame is float. So we need
                                         # to convert it to integer before it can be used as indexing.
            histo[pixel] += 1               
    probability = histo/_totalpixel    # Convert histogram bins to probability
    meangray = 0
    for i in range(256):
        meangray += probability[i]*i   # Calculate mean gray level.
    
    diffgray = meangray - 150          # Difference between desired mean leval and current gray level.
    
    for i in range(_ROI_STARTY,_ROI_STARTY+_ROI_HEIGHT):      # Remap the gray scale level in the original image
        for j in range(_ROI_STARTX,_ROI_STARTX+_ROI_WIDTH):   # into the equalized image. To reduce computation,
            pixel = imgIresize[i,j]                           # we only carry out the remapping on the ROI.
            imgIresize[i,j] = pixel - diffgray
    #--- End of Perform Equalization of Gray Level Mean ---                                                           
                                                        
    # Crop out region-of-interest (ROI)
    imggrayresizecrop = imgIresize[_ROI_STARTY:_ROI_STARTY+_ROI_HEIGHT,_ROI_STARTX:_ROI_STARTX+_ROI_WIDTH] 
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
        pwmLED1.value = _LED_MAX_BRIGHTNESS
        pygame.draw.rect(screen,(255,255,0),pgrectL1,width=rescale_level)
    elif result == 2:
        pwmLED1.value = _LED_MAX_BRIGHTNESS
        pygame.draw.rect(screen,(255,255,0),pgrectL2,width=rescale_level)
    elif result == 3:
        pwmLED1.value = _LED_MAX_BRIGHTNESS
        pygame.draw.rect(screen,(255,255,0),pgrectL3,width=rescale_level)
    elif result == 4:
        pwmLED1.value = _LED_MAX_BRIGHTNESS
        pygame.draw.rect(screen,(255,255,0),pgrectL4,width=rescale_level)
    else:
        pwmLED1.value = _LED_LOW_BRIGHTNESS     # Turn on LED at low brigthness.
    HeadRC_Interface.sendtoRC(bytes([82,49,48+result,48,48,10])) #"R1x00", where x = '0','1',...'4' 
                
    display.update()                                    # This will create a window and display the image.
    #display.flip()
    for event in pygame.event.get():  # Just close the window and a QUIT even will be generated
        if event.type == pygame.QUIT:
            is_running = False

# --- House keeping tasks when exiting ---
HeadRC_Interface.stop()         # Stop all thread
mycam.stop()
pygame.quit()
#pwmLED1.off()                  # Turn off LED to indicate that program is no longer executing.
if sport.is_open == True:       # Make sure to close the serial port if it is opened, otherwise other
    sport.close()               # process cannot use the serial port.