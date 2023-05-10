import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
import numpy as np    # for mathematical operations
import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm
import os

import os
import cv2
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from moviepy.editor import *
from collections import deque
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

image_height, image_width = 128, 128
max_images_per_class = 800

dataset_directory = '/Users/parthkalathia/Desktop/CNN_SCRATCH/FINALDATASET_V2'

num_class = 5
# classes_list = ['TALKING', 'HEADDOWN', 'YAWNING', 'MOBILE', 'NORMAL']
classes_list = ['TALKING']
model_output_size = len(classes_list)

def frames_extraction(video_path, num_of_fps):
    
    # Empty List declared to store video frames
    frames_list = []
    count = 0
    # Reading the Video File Using the VideoCapture
    video_reader = cv2.VideoCapture(video_path)
    
    # Get the frames per second (fps) of the video
    fps = int(video_reader.get(cv2.CAP_PROP_FPS))
    
    # Set the interval between frames to extract
    if fps > num_of_fps:
        frame_interval = int(fps / num_of_fps)
    else:
        frame_interval = fps
    
    # Initialize frame counter
    frame_count = 0
    # Iterating through Video Frames
    while True:
          
        # Reading a frame from the video file 
        success, frame = video_reader.read()
        
        # If Video frame was not successfully read then break the loop
        if not success:
            break
    
        # Increment the frame counter
        frame_count += 1
        
        if frame_count % frame_interval == 0:
            # Resize the Frame to fixed Dimensions
            resized_frame = cv2.resize(frame, (image_height, image_width))
        
            # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
            normalized_frame = resized_frame / 255
        
            # Appending the normalized frame into the frames list
            frames_list.append(normalized_frame)
            
            count += 1
#             print(f"{count} done")
        
    # Closing the VideoCapture object and releasing all resources. 
    video_reader.release()
    
    return frames_list


def create_dataset(num_of_fps):
    import random
    # Declaring Empty Lists to store the features and labels values.
    temp_features = [] 
    features = []
    labels = []
    # Iterating through all the classes mentioned in the classes list
    for class_index, class_name in enumerate(classes_list):
        print(f'Extracting Data of Class: {class_name}')
        
        # Getting the list of video files present in the specific class name directory
        files_list = os.listdir(os.path.join(dataset_directory, class_name))
        print(files_list)

        # Iterating through all the files present in the files list
        for file_name in files_list:
            
            # Construct the complete video path
            video_file_path = os.path.join(dataset_directory, class_name, file_name)
            print(video_file_path)

            # Calling the frame_extraction method for every video file path
            frames = frames_extraction(video_file_path, num_of_fps)
          

            # Appending the frames to a temporary list.
            temp_features.extend(frames)
        
        # Adding randomly selected frames to the features list
        features.extend(random.sample(temp_features, max_images_per_class))

        # Adding Fixed number of labels to the labels list
        labels.extend([class_index] * max_images_per_class)
        
        # Emptying the temp_features list so it can be reused to store all frames of the next class.
        temp_features.clear()

    # Converting the features and labels lists to numpy arrays
    features = np.asarray(features)
    labels = np.array(labels) 
    
    return features, labels  


def plot_metric(metric_name_1, metric_name_2, plot_name, model_name):
  # Get Metric values using metric names as identifiers
  metric_value_1 = model_name.history[metric_name_1]
  metric_value_2 = model_name.history[metric_name_2]

  # Constructing a range object which will be used as time 
  epochs = range(len(metric_value_1))
  
  # Plotting the Graph
  plt.plot(epochs, metric_value_1, 'blue', label = metric_name_1)
  plt.plot(epochs, metric_value_2, 'red', label = metric_name_2)
  
  # Adding title to the plot
  plt.title(str(plot_name))

  # Adding legend to the plot
  plt.legend()


def predict_on_live_video(video_file_path, output_file_path, window_size, model):
   #  import imutils
    import cv2
    from collections import deque

    # Initialize a Deque Object with a fixed size which will be used to implement moving/rolling average functionality.
    predicted_labels_probabilities_deque = deque(maxlen = window_size)

    # Reading the Video File using the VideoCapture Object
    video_reader = cv2.VideoCapture(video_file_path)

    # Getting the width and height of the video 
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writing the Overlayed Video Files Using the VideoWriter Object
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 24, (original_video_width, original_video_height))
    image_height=64
    image_width =64
    while True: 

        # Reading The Frame
        status, frame = video_reader.read() 

        if not status:
            break
        print(image_width)
        # Resize the Frame to fixed Dimensions
       
        resized_frame = cv2.resize(frame, (image_height, image_width))
        print(resized_frame.shape)
        resized_frame= resized_frame.reshape(-1, 16, 16,3)
        print(resized_frame.shape)
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = np.array(resized_frame) / 255

        # Passing the Image Normalized Frame to the model and receiving Predicted Probabilities.
        predicted_labels_probabilities = model.predict(np.expand_dims(normalized_frame, axis = 0))[0]
        
      
        # Appending predicted label probabilities to the deque object
        predicted_labels_probabilities_deque.append(predicted_labels_probabilities)

        #Assuring that the Deque is completely filled before starting the averaging process
        if len(predicted_labels_probabilities_deque) == window_size:

           # Converting Predicted Labels Probabilities Deque into Numpy array
            predicted_labels_probabilities_np = np.array(predicted_labels_probabilities_deque)

          #  Calculating Average of Predicted Labels Probabilities Column Wise 
            predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)

            # Converting the predicted probabilities into labels by returning the index of the maximum value.
            predicted_label = np.argmax(predicted_labels_probabilities_averaged)
            

            # Accessing The Class Name using predicted label.
            predicted_class_name = classes_list[predicted_label]
            print(predicted_label)
          
            # Overlaying Class Name Text Ontop of the Frame
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Writing The Frame
        video_writer.write(frame)


        cv2.imshow('Predicted Frames', frame)

        # key_pressed = cv2.waitKey(10)

        # if key_pressed == ord('q'):
        #     break

    # cv2.destroyAllWindows()

    
    # Closing the VideoCapture and VideoWriter objects and releasing all resources held by them. 
    video_reader.release()
    video_writer.release()


def make_average_predictions(video_file_path, predictions_frames_count, model):
    
    # Initializing the Numpy array which will store Prediction Probabilities
    predicted_labels_probabilities_np = np.zeros((predictions_frames_count, model_output_size), dtype = np.float)

    # Reading the Video File using the VideoCapture Object
    video_reader = cv2.VideoCapture(video_file_path)
    print(video_file_path)
    # Getting The Total Frames present in the video 
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculating The Number of Frames to skip Before reading a frame
    skip_frames_window = video_frames_count // predictions_frames_count
    
    all_frame_predictions_proability = {}
    for frame_counter in range(predictions_frames_count): 

        # Setting Frame Position
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Reading The Frame
        _ , frame = video_reader.read() 

        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (image_height, image_width))
        resized_frame= resized_frame.reshape(-1, image_height, image_width,3)
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255

        # Passing the Image Normalized Frame to the model and receiving Predicted Probabilities.
        predicted_labels_probabilities = model.predict(np.expand_dims(normalized_frame, axis = 0))[0]

        # Appending predicted label probabilities to the deque object
        predicted_labels_probabilities_np[frame_counter] = predicted_labels_probabilities

        all_frame_predictions_proability[frame_counter] = predicted_labels_probabilities
        
    # Calculating Average of Predicted Labels Probabilities Column Wise 
    predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)

    # Sorting the Averaged Predicted Labels Probabilities
    predicted_labels_probabilities_averaged_sorted_indexes = np.argsort(predicted_labels_probabilities_averaged)[::-1]

    # class_name = []
    # predicted_class_prebability = []
    predicted_dict = {}
    # Iterating Over All Averaged Predicted Label Probabilities
    for predicted_label in predicted_labels_probabilities_averaged_sorted_indexes:

        # Accessing The Class Name using predicted label.
        predicted_class_name = classes_list[predicted_label]

        # Accessing The Averaged Probability using predicted label.
        predicted_probability = predicted_labels_probabilities_averaged[predicted_label]
        predicted_dict[predicted_class_name] = predicted_probability
        
        print(predicted_probability)
       # print(f"CLASS NAME: {predicted_class_name}   AVERAGED PROBABILITY: {(predicted_probability*100):.2}")
    
    # Closing the VideoCapture Object and releasing all resources held by it. 
    video_reader.release()

    return predicted_dict, all_frame_predictions_proability