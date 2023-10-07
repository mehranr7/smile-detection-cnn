import os
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from Extract_Parts import Face_detector, Cascade_model
from Model_Manager import LeNet
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

class Model_Trainer:
    # define folder paths
    source_folder_path = 'genki4k'
    faces_folder_path = os.path.join(source_folder_path,"faces")
    smiles_folder_path = os.path.join(source_folder_path,"smiles")
    width = 28
    height = 14

    @staticmethod
    # read genki4 lables file
    def Read_labels():
        f = open("labels.txt", "r")
        label = np.empty(0)
        row = f.readline()
        while row != "":
            col = row.split()
            label = np.append(label,col[0])
            row = f.readline()
        return label

    @staticmethod
    # get index of a sample
    def Get_sample_index(filename):
        file_part = filename.split("-")[0]
        number_part = file_part[4:]
        return int(number_part) - 1

    @staticmethod
    def Train(model_name):
        # define folder paths
        source_folder_path = 'genki4k'
        faces_folder_path = os.path.join(source_folder_path,"faces")
        smiles_folder_path = os.path.join(source_folder_path,"smiles")

        # detect faces
        Face_detector.part_croper(source_folder_path, faces_folder_path, Cascade_model.Face.value)

        # detect smilse
        Face_detector.part_croper(faces_folder_path, smiles_folder_path, Cascade_model.Smile.value, True, Model_Trainer.height, Model_Trainer.width)

        # get list of samples
        samples = os.listdir(smiles_folder_path)
        labels_files = Model_Trainer.Read_labels()

        data = []
        labels = []

        # collect data from samples
        for image in samples:
            img = cv2.imread(os.path.join(smiles_folder_path,image))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = img_to_array(gray)
            data.append(gray)
            index = Model_Trainer.Get_sample_index(image)
            labels.append(labels_files[index])

        # convert the labels from integers to vectors
        le = LabelEncoder().fit(labels)
        labels = to_categorical(le.transform(labels), 2)

        # convert to numpy array and make it between 0 to 1
        data = np.array(data, dtype = "float") / 255.0

        # get model ready
        model = LeNet.build(1, Model_Trainer.width, Model_Trainer.height, 2)
        model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

        # partition the data into training and testing splits using 80% of the data
        # for training and remaining 20% for testing
        (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.2, stratify = labels, random_state = 42)

        # initialize the model
        print("[INFO] compiling model...")
        model = LeNet.build(numChannels = 1, imgRows = Model_Trainer.height, imgCols = Model_Trainer.width, numClasses = 2)
        model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

        # print the summary of the model
        model.summary()
        
        # train the network
        print("[INFO] training network...")
        H = model.fit(trainX, trainY, validation_data = (testX, testY), batch_size = 64, epochs = 15, verbose = 1)

        # evaluate the network
        print("[INFO] evaluating network...")
        predictions = model.predict(testX, batch_size = 64)
        print(classification_report(testY.argmax(axis = 1), predictions.argmax(axis = 1), target_names = le.classes_))

        # save the model to disk
        model.save(model_name)