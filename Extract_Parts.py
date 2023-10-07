import os
import cv2
from enum import Enum
import numpy as np

# define opencv cascade pre-trained models
class Cascade_model(Enum):
    class Face(Enum):
        Name = "face"
        Model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        Scale_factor = 1.3
        Min_Neighbors = 5
    class Smile(Enum):
        Name = "smile"
        Model = cv2.CascadeClassifier('haarcascade_smile.xml')
        Scale_factor = 1.4
        Min_Neighbors = 8

class Face_detector:
    @staticmethod
    # get best size based on aspect ratio
    def get_size_ratio(min_width, min_height, ratio = (4,3)):
        if ratio[0] > ratio[1]:
            if min_width > min_height:
                min_height = min_width * ratio[1] / ratio[0]
            else:
                min_width = min_height * ratio[0] / ratio[1]
        else:
            if min_width > min_height:
                min_height = min_width * ratio[1] / ratio[0]
            else:
                min_width = min_height * ratio[0] / ratio[1]
        return round(min_width), round(min_height)

    @staticmethod
    # crop parts from images and save in another folder
    def part_croper(input_folder_path, target_folder_path, cascade_model, should_resize = False, height = 128, width = 128):
        images = os.listdir(input_folder_path)

        # remove old files
        files_to_remove = os.listdir(target_folder_path)
        if len(files_to_remove) > 0:
            for file in files_to_remove:
                os.remove(os.path.join(target_folder_path,file))

        if len(images) > 0:
            for image in images:
                if not ".jpg" in image.lower():
                    continue
                
                # read image
                img = cv2.imread(os.path.join(input_folder_path,image))

                # detect parts
                parts, locations = Face_detector.crop_photo_parts(img, cascade_model, should_resize, height, width)
                
                # save detected parts
                if len(parts) > 0:
                    for i in range(len(parts)):
                        file_name = os.path.join(target_folder_path, f'{image[:-4]}-{cascade_model.Name.value}{i+1}.jpg')
                        if os.path.exists(file_name):
                            os.remove(file_name)
                        cv2.imwrite(file_name, parts[i])
                        print(file_name)
                else :
                    print(image,f"No {cascade_model.Name.value} detected!")

                # If it doesn't exist, create the target folder
                if not os.path.exists(target_folder_path):
                    os.makedirs(target_folder_path)         
    
    @staticmethod
    # corp an image
    def crop_photo_parts(image, cascade_model, should_resize = False, height = 128, width = 128):

        # detects parts in the input image
        parts = cascade_model.Model.value.detectMultiScale(image,
        cascade_model.Scale_factor.value,
        cascade_model.Min_Neighbors.value)
        
        cropped_parts = []
        locations = []

        # save detected parts
        if len(parts) > 0:
            for (x, y, w, h) in parts:

                # get the best width and height for being ready to resize
                if should_resize:
                    w, h = Face_detector.get_size_ratio(w,h, (width, height))
                part = image[y:y + h, x:x + w]

                # resize the photo for consistency
                if should_resize:
                    part = cv2.resize(part, (width, height), interpolation=cv2.INTER_LINEAR)

                # store parts and locations
                cropped_parts.append(part)
                locations.append((x, y, w, h))
        
        return cropped_parts , locations