import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import os



points = [1, #noise
          300, 293, 334, 296, 336, 285, 295, 282, 283, #left eyebrow
          70, 63, 105, 66, 107, 55, 65, 52, 53,  #right eyebrow
          61, 185, 40, 39, 37, 0, 267, 269, 270, 409, #lipsUpperOuter
          291, 375, 321, 405, 314, 17, 84, 181, 91, 146, #lipsLowerOuter
          80, 81, 82, 13, 312, 311, 310, #lipsUpperInner
          308, 318, 402, 317, 14, 87, 178, 88, 78, #lipsLowerInner
          33,161, 160, 159, 158, 157, #right eye upper 0 
          133, 154, 153, 145, 144, 163,  #right eye lower 0 
          362, 384, 385, 386, 387, 388, #leftEyeUpper0
          263, 390, 373, 374, 380, 381 #leftEyeLower0 
    ]


def min_max_scaling(coordinates):
    min_values = np.min(coordinates, axis=0)
    max_values = np.max(coordinates, axis=0)
    
    scaled_coordinates = (coordinates - min_values) / (max_values - min_values)
    
    return scaled_coordinates



def calculate_euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

def calculate_distances(points):
    num_points = len(points)
    distances = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(num_points):
            distances[i, j] = calculate_euclidean_distance(points[i], points[j])

    return distances


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh



def read_images_from_folder(folder_path):
    df = pd.DataFrame(columns=['eyebrows', 'lips', 'eyes'])
    counter_img = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
       
            input_image = cv2.imread(file_path)
            
            with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5) as face_mesh:

                image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)

                if results.multi_face_landmarks:
                    
                    for face_landmarks in results.multi_face_landmarks:
                        k=0
                        values= []
                        for index in points:
                            landmark = face_landmarks.landmark[index]
                            x = int(landmark.x * image_rgb.shape[1])
                            y = int(landmark.y * image_rgb.shape[0])
                           
                            values.append([x,y])
                            k=k+1
                
                    scaled_values = min_max_scaling(values)
                    # Euclidean distance from nose tip
                    distances = calculate_distances(scaled_values)
                    first_point_distances = distances[0, 1:]

                    
                    for_eyebrow = np.mean(first_point_distances[:18])
                    for_lips = np.mean(first_point_distances[18:54])
                    for_eyes = np.mean(first_point_distances[54:])
                    df.loc[counter_img] = [for_eyebrow, for_lips, for_eyes]
                    counter_img +=1
                        
                
               
    print(f"Number of images = {counter_img}")
    return df


def facial_landmark(frame):
    test_df = pd.DataFrame(columns=['eyebrows', 'lips', 'eyes'])
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                
                for face_landmarks in results.multi_face_landmarks:
                    k=0
                    values= []
                    for index in points:
                        landmark = face_landmarks.landmark[index]
                        x = int(landmark.x * image_rgb.shape[1])
                        y = int(landmark.y * image_rgb.shape[0])
                       
                        values.append([x,y])
                        k=k+1
            
                scaled_values = min_max_scaling(values)
                # Euclidean distance from nose tip
                distances = calculate_distances(scaled_values)
                first_point_distances = distances[0, 1:]

                
                for_eyebrow = np.mean(first_point_distances[:18])
                for_lips = np.mean(first_point_distances[18:54])
                for_eyes = np.mean(first_point_distances[54:])
                test_df.loc[0] = [for_eyebrow, for_lips, for_eyes]
                    
    
    return test_df
    


happy = read_images_from_folder("dataset/happy")
angry = read_images_from_folder("dataset/angry")
sad = read_images_from_folder("dataset/sad")
fair = read_images_from_folder("dataset/fear")
neutral = read_images_from_folder("dataset/neutral")
suprised = read_images_from_folder("dataset/surprised")


def createDF(variable):
    df_variable = pd.DataFrame({'min': [np.min(happy[variable]), np.min(angry[variable]), np.min(sad[variable]), np.min(fair[variable]), np.min(neutral[variable]), np.min(suprised[variable])],
                               'max': [np.max(happy[variable]), np.max(angry[variable]), np.max(sad[variable]), np.max(fair[variable]), np.max(neutral[variable]), np.max(suprised[variable])],
                               'median': [np.median(happy[variable]), np.median(angry[variable]), np.median(sad[variable]), np.median(fair[variable]), np.median(neutral[variable]), np.median(suprised[variable])],
                               'mean': [np.mean(happy[variable]), np.mean(angry[variable]), np.mean(sad[variable]), np.mean(fair[variable]), np.mean(neutral[variable]), np.mean(suprised[variable])]
                               })

    df_variable.index = ['happy', 'angry', 'sad', 'fair','neutral','suprised']
    return df_variable

df_eyebrow = createDF("eyebrows")
df_lips = createDF("lips")
df_eyes = createDF("eyes")