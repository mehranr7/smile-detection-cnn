import tensorflow as tf
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from Extract_Parts import Face_detector, Cascade_model
from Trainer import Model_Trainer

# detect smilse 
def detect_smile(frame, model):

    # detect faces
    faces, face_locations = Face_detector.crop_photo_parts(frame, Cascade_model.Face.value)
    for i in range(len(faces)):
        # get list of smiles
        smile_list, loacions = Face_detector.crop_photo_parts(faces[i], Cascade_model.Smile.value, True, Model_Trainer.height, Model_Trainer.width)

        # get a clone of previous frame and count smiles
        clone_frame = frame
        j = 0

        if len(smile_list) > 0:
            
            # detect each sample of smile_list
            for smile in smile_list:
                gray = cv2.cvtColor(smile, cv2.COLOR_BGR2GRAY)
                sample = img_to_array(gray)
                sample = np.array(sample, dtype = "float") / 255.0
                data = []
                data.append(sample)
                data = np.array(data)

                # predict using model and find the best match
                predictions = model.predict(data)
                label = "Smiling" if predictions[0][1] > predictions[0][0] else "Not Smiling"

                # asume the last frame if it isn't first sample
                if j != 0:
                    frame = clone_frame

                # put related result to the frame
                (x, y, w, h) = face_locations[i]
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
                
                j = j + 1
    return frame

class Smile_Detector:
    @staticmethod
    def Detect_Video(file_name,model_name):
        # Load the saved model
        model = tf.keras.models.load_model(model_name)
        # Open the video file
        cap = cv2.VideoCapture(file_name)

        # Loop over all frames in the video file
        while cap.isOpened():
            # Read the next frame from the video file
            ret, frame = cap.read()

            # if 'q' key is pressed, stop the loop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # If there are no more frames, exit the loop
            if not ret:
                break

            # show our detected faces along with smiling/not smiling labels
            frame = detect_smile(frame, model)
            cv2.imshow("video", frame)