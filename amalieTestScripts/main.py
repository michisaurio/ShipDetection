"""
Hentet fra: https://github.com/gurkandemir/Bag-of-Visual-Words

IMAGE CLASSIFICATION
"""

import argparse
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import accuracy_score
import sklearn.metrics as metric
from xml.etree import ElementTree as et
import time
import pickle


def getFilePath(folder):
    """
    Hent ut (returner) path til alle bildefilene og .xml filene
    :param folder:
    :return: array with filepaths: "dataset/Optisk/190616_112733.jpg/.xml"
    """
    file_path = []
    directory = os.listdir(folder)
    print(folder)
    for filename in directory:
        path = folder + "/" + filename  # "dataset/Optisk/190616_112733.jpg"
        file_path.append(path)  # Legg til alle biler dataset/train/img_name.jpg

    # array[start_index:end_index:step]
    image_path = file_path[0::2]
    annotation_path = file_path[1::2]

    #print(image_path[0])
    print("Length image path:", len(image_path))
    #print(annotation_path[0])
    print("Length annotation path:", len(annotation_path))

    return image_path, annotation_path

def readImageInFolder(img_path):
    """
    Read all images in the dataset folder (img_path). Resize to (400,300)
    :param img_path: The path to the image
    :return: The resized images
    """
    # img = cv2.imread('dataset/train/ship1.jpg')
    images = []
    for path in img_path:
        img = cv2.imread(path)   # Read image and convert to grayscale
        #img_resize = cv2.resize(img, (400, 300))
        #images.append(img_resize)
        images.append(img)
    return images

def scaleImageFromROI(images, label_path):
    """

    :param images:
    :param label_path:
    :return:
    """
    cropped_images = []


    for i in range(len(images)):
        tree = et.parse(label_path[i])
        myroot = tree.getroot()

        tag_name = ['xmin', 'ymin', 'xmax', 'ymax']

        tag = 0
        for x in myroot.findall('object'):
            name = x.find('name')


            if "motorboat with priority" in name.text:
                #print("In motorboat")
                xmin = x.find('bndbox/' + tag_name[0])
                #print(xmin.text)
                ymin = x.find('bndbox/' + tag_name[1])
                #print(ymin.text)
                xmax = x.find('bndbox/' + tag_name[2])
                #print(xmax.text)
                ymax = x.find('bndbox/' + tag_name[3])
                #print(ymax.text)
                tag += 1
                #print(tag)
                img = images[i]
                ROI = img[int(float(ymin.text)):int(float(ymax.text)), int(float(xmin.text)):int(float(xmax.text))]
                # Save image to "result"-folder
                dir = 'Cropped/motorboat_priority'
                cv2.imwrite(os.path.join(dir, "motorboat_priority" + str(i) + "_" + str(tag) + ".jpg"), ROI)
                cv2.waitKey(0)
                cropped_images.append(ROI)

            elif "sailboat with sails down" in name.text:
                #print("In sailboat")
                xmin = x.find('bndbox/' + tag_name[0])
                #print(xmin.text)
                ymin = x.find('bndbox/' + tag_name[1])
                #print(ymin.text)
                xmax = x.find('bndbox/' + tag_name[2])
                #print(xmax.text)
                ymax = x.find('bndbox/' + tag_name[3])
                #print(ymax.text)

                tag += 1
                #print(tag)

                img = images[i]
                ROI = img[int(float(ymin.text)):int(float(ymax.text)), int(float(xmin.text)):int(float(xmax.text))]
                # Save image to "result"-folder
                dir = 'Cropped/sailboat_down'
                cv2.imwrite(os.path.join(dir, "sailboat_down" + str(i) + "_" + str(tag) + ".jpg"), ROI)
                cv2.waitKey(0)
                cropped_images.append(ROI)

            elif "sailboat with sails up" in name.text:
                #print("In sailboat")
                xmin = x.find('bndbox/' + tag_name[0])
                #print(xmin.text)
                ymin = x.find('bndbox/' + tag_name[1])
                #print(ymin.text)
                xmax = x.find('bndbox/' + tag_name[2])
                #print(xmax.text)
                ymax = x.find('bndbox/' + tag_name[3])
                #print(ymax.text)

                tag += 1
                #print(tag)

                img = images[i]
                ROI = img[int(float(ymin.text)):int(float(ymax.text)), int(float(xmin.text)):int(float(xmax.text))]
                # Save image to "result"-folder
                dir = 'Cropped/sailboat_up'
                cv2.imwrite(os.path.join(dir, "sailboat_up" + str(i) + "_" + str(tag) + ".jpg"), ROI)
                cv2.waitKey(0)
                cropped_images.append(ROI)

            elif "barge" in name.text:
                #print("In barge")
                xmin = x.find('bndbox/' + tag_name[0])
                #print(xmin.text)
                ymin = x.find('bndbox/' + tag_name[1])
                #print(ymin.text)
                xmax = x.find('bndbox/' + tag_name[2])
                #print(xmax.text)
                ymax = x.find('bndbox/' + tag_name[3])
                #print(ymax.text)

                tag += 1
                #print(tag)

                img = images[i]
                ROI = img[int(float(ymin.text)):int(float(ymax.text)), int(float(xmin.text)):int(float(xmax.text))]
                # Save image to "result"-folder
                dir = 'Cropped/barge'
                cv2.imwrite(os.path.join(dir, "barge" + str(i) + "_" + str(tag) + ".jpg"), ROI)
                cv2.waitKey(0)
                cropped_images.append(ROI)

            elif "building" in name.text:
                var = None
            elif "front" in name.text:
                var = None
            elif "back" in name.text:
                var = None
            elif "side" in name.text:
                var = None
            elif "wake" in name.text:
                var = None
            elif "mast" in name.text:
                var = None
            elif "overbygg" in name.text:
                var = None
            elif "bridge" in name.text:
                var = None
            else: # motorboat
                # print("In motorboat")
                xmin = x.find('bndbox/' + tag_name[0])
                # print(xmin.text)
                ymin = x.find('bndbox/' + tag_name[1])
                # print(ymin.text)
                xmax = x.find('bndbox/' + tag_name[2])
                # print(xmax.text)
                ymax = x.find('bndbox/' + tag_name[3])
                # print(ymax.text)
                tag += 1
                # print(tag)
                img = images[i]
                ROI = img[int(float(ymin.text)):int(float(ymax.text)), int(float(xmin.text)):int(float(xmax.text))]
                # Save image to "result"-folder
                dir = 'Cropped/motorboat'
                cv2.imwrite(os.path.join(dir, "motorboat" + str(i) + "_" + str(tag) + ".jpg"), ROI)
                cv2.waitKey(0)
                cropped_images.append(ROI)
    return cropped_images

def splitDataset():
    # TODO split dataset with cropped images into training and testing
    # Get the classified images
    motorboat_priority = getClassImages('dataset/cropped_images/motorboat_priority')
    sailboat_down = getClassImages('dataset/cropped_images/sailboat_down')
    sailboat_up = getClassImages('dataset/cropped_images/sailboat_up')
    barge = getClassImages('dataset/cropped_images/barge')
    motorboat = getClassImages('dataset/cropped_images/motorboat')


    dataset = [motorboat_priority, sailboat_down, sailboat_up, barge, motorboat]

    train = []
    test = []
    for i in dataset:
        train_i, test_i = train_test_split(i, test_size=0.2, train_size=0.8, shuffle=True)
        #print("Length train: ", len(train_i))
        #print("Length test: ", len(test_i))
        #print("")

        train.append(train_i)
        test.append(test_i)

    train_dict = {
        "motorboat_priority": train[0],  # motorboat_priority
        "sailboat_down": train[1],  # sailboat_down
        "sailboat_up": train[2],  # sailboat_up
        "barge": train[3],  # barge
        "motorboat": train[4]   # motorboat
    }

    test_dict = {
        "motorboat_priority": test[0],  # motorboat_priority
        "sailboat_down": test[1],  # sailboat_down
        "sailboat_up": test[2],  # sailboat_up
        "barge": test[3],  # barge
        "motorboat": test[4]   # motorboat
    }
    return train_dict, test_dict


def getClassImages(path):
    #path = 'Cropped/barge'
    class_images = []
    file_path = []
    for filename in os.listdir(path):
        n_path = path + "/" + filename  # "dataset/Optisk/190616_112733.jpg"
        file_path.append(n_path)
        img = cv2.imread(n_path)  # Read image and convert to grayscale
        class_images.append(img)
    return class_images


def getDescriptors(sift, orb, extractor, img):
    #kp, des = sift.detectAndCompute(img, None)
    if "sift" in extractor:# Funker
        #sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)

    if "orb" in extractor: # Funker
        #orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(img, None)


    img_kp = cv2.drawKeypoints(img, kp, None, color=(0, 0, 255))
    #cv2.imwrite('img_keypoint.png', img_kp)
    return des, img_kp


def readImage(img_path):
    #img = cv2.imread(img_path, 0)
    img = cv2.imread(img_path, 0)
    return img  # cv2.resize(img, (150, 150))


def vstackDescriptors(descriptor_list):
    descriptors = np.array(descriptor_list[0])
    #print("In vstackDescriptor - Shape descriptor: ", np.shape(descriptors))
    for descriptor in descriptor_list[1:]:
        #print("Shape descriptor: ", np.shape(descriptor))
        descriptors = np.vstack((descriptors, descriptor))
    return descriptors


def clusterDescriptors(descriptors, no_clusters):
    #kmeans = KMeans(init="random", n_clusters=no_clusters, n_init=10, max_iter=300, random_state=42)
    #kmeans.fit(descriptors)
    kmeans = KMeans(n_clusters=no_clusters).fit(descriptors) # TODO legg til init='k-means++'
    return kmeans


def extractFeatures(kmeans, descriptor_list, image_count, no_clusters):
    im_features = np.array([np.zeros(no_clusters) for i in range(image_count)])
    for i in range(image_count):
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
            feature = feature.reshape(1, 128)
            idx = kmeans.predict(feature)
            im_features[i][idx] += 1
    return im_features


def normalizeFeatures(scale, features):
    return scale.transform(features)


def plotHistogram(im_features, no_clusters):
    x_scalar = np.arange(no_clusters)
    y_scalar = np.array([abs(np.sum(im_features[:, h], dtype=np.int32)) for h in range(no_clusters)])

    plt.bar(x_scalar, y_scalar)
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.title("Complete Vocabulary Generated")
    plt.xticks(x_scalar + 0.4, x_scalar)
    plt.show()


def svcParamSelection(X, y, kernel, nfolds):
    Cs = [0.5, 0.1, 0.15, 0.2, 0.3]
    gammas = [0.1, 0.11, 0.095, 0.105]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(SVC(kernel=kernel), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


def findSVM(im_features, train_labels, kernel):
    features = im_features
    if (kernel == "precomputed"):
        features = np.dot(im_features, im_features.T)

    params = svcParamSelection(features, train_labels, kernel, 5)
    C_param, gamma_param = params.get("C"), params.get("gamma")
    print(C_param, gamma_param)
    class_weight = {
        0: (807 / (5 * 140)),
        1: (807 / (5 * 140)),
        2: (807 / (5 * 133)),
        3: (807 / (5 * 133)),
        4: (807 / (5 * 133))
    }

    svm = SVC(kernel=kernel, C=C_param, gamma=gamma_param, class_weight=class_weight)
    svm.fit(features, train_labels)
    return svm


def plotConfusionMatrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def plotConfusions(true, predictions):
    np.set_printoptions(precision=2)

    class_names = ["motorboat_priority", "sailboat_down", "sailboat_up", "barge", "motorboat"]

    plotConfusionMatrix(true, predictions, classes=class_names,
                        title='Confusion matrix, without normalization')

    plotConfusionMatrix(true, predictions, classes=class_names, normalize=True,
                        title='Normalized confusion matrix')
    plt.show()


def findAccuracy(true, predictions):
    print('accuracy score: %0.3f' % metric.accuracy_score(true, predictions))

    #print('f1 score: %0.3f' % metrics.f1_score(true, predictions))
    #print('precision score: %0.3f' % metrics.precision_score(true, predictions))
    #print('recall score: %0.3f' % metrics.recall_score(true, predictions))


# TODO train model
def trainModel(train_dict, no_clusters, kernel):
    time1 = time.time()


    # Create extractors
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0,
                         WTA_K=2, patchSize=31, fastThreshold=10)

    descriptor_list = []
    kp_img_list = []
    zero_des_list = []
    train_labels = np.array([])
    count = 0

    train_images = []
    for k, img_list in train_dict.items():
        for img in img_list:
            des, kp_img = getDescriptors(sift, orb, "sift", img)
            train_images.append(img)

            if des is not None:
                count += 1
                descriptor_list.append(des)
                kp_img_list.append(kp_img)
                #print("We are looking at a " + k)
                if k == "motorboat_priority":
                    class_index = 0
                if k == "sailboat_down":
                    class_index = 1
                if k == "sailboat_up":
                    class_index = 2
                if k == "barge":
                    class_index = 3
                if k == "motorboat":
                    class_index = 4
                train_labels = np.append(train_labels, class_index)
            else:
                print("Found zero-descriptor in training set")

    print("Nr. of images", len(train_images))
    print("Train labels:", len(train_labels))

    print("Length/size descriptor list: ", len(descriptor_list), np.shape(descriptor_list))
    # print(np.shape(descriptor_list[9]))
    # print(np.shape(descriptor_list[10]))
    print("Lenght descriptor 0: ", len(descriptor_list[0]), np.shape(descriptor_list[0]))
    print("Lenght descriptor 10: ", len(descriptor_list[10]), np.shape(descriptor_list[10]))

    #plt.imshow(kp_img_list[10], cmap=plt.cm.gray)
    #plt.axis('off')
    #plt.title("Image")
    #plt.show()
    # print("train labels:", train_labels)

    descriptors = vstackDescriptors(descriptor_list)
    print("Descriptors vstacked.")

    kmeans = clusterDescriptors(descriptors, no_clusters)
    print("Descriptors clustered.")

    # image_count = len(image_path) - len(zero_des_list) # Subtract the zero-descriptors
    im_features = extractFeatures(kmeans, descriptor_list, count, no_clusters)
    print("Images features extracted.")

    scale = StandardScaler().fit(im_features)
    im_features = scale.transform(im_features)
    print("Train images normalized.")

    plotHistogram(im_features, no_clusters)
    print("Features histogram plotted.")

    #svm = findSVM(im_features, train_labels, kernel)
    #print(im_features)
    svm = SVC(kernel=kernel) # Initialize the Support Vector Classifier (SVC)
    svm.fit(im_features, train_labels) # Fit the model
    print("SVM fitted.")

    # Make predictions
    predictions = svm.predict(im_features)
    print(predictions)
    print("Training completed.")
    print("")

    time2 = time.time()
    print("... Running training time:", (time2 - time1) / 60, "min ...")

    return kmeans, scale, svm, im_features


# TODO test model
def testModel(test_dict, kmeans, scale, svm, im_features, no_clusters, kernel):
    time1 = time.time()

    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0,
                         WTA_K=2, patchSize=31, fastThreshold=10)

    count = 0
    true = []
    descriptor_list = []

    name_dict = {
        "0": "motorboat_priority",
        "1": "sailboat_down",
        "2": "sailboat_up",
        "3": "barge",
        "4": "motorboat"
    }


    #for path in test_images: # TODO bytt ut path med test images
    for k, img_list in test_dict.items():
        for img in img_list:
            #img = readImage(path)
            des, _ = getDescriptors(sift, orb, "sift", img)

            if (des is not None):
                count += 1
                descriptor_list.append(des)

                if k == "motorboat_priority":
                    true.append("motorboat_priority")
                elif k == "sailboat_down":
                    true.append("sailboat_down")
                elif k == "sailboat_up":
                    true.append("sailboat_up")
                elif k == "barge":
                    true.append("barge")
                else:
                    true.append("motorboat")
            else:
                print("Found zero-descriptor in test set")


    #print("descriptor_list shape: ", np.shape(descriptor_list))
    #print("descriptor_list[0] shape: ", np.shape(descriptor_list[0]))
    #print("descriptor_list[10] shape: ", np.shape(descriptor_list[10]))

    #descriptors = vstackDescriptors(descriptor_list)
    print("")
    #print("descriptors shape: ", np.shape(descriptors))

    test_features = extractFeatures(kmeans, descriptor_list, count, no_clusters)

    test_features = scale.transform(test_features)
    print("test features:", len(test_features))

    kernel_test = test_features
    if (kernel == "precomputed"):
        kernel_test = np.dot(test_features, im_features.T)

    predictions = [name_dict[str(int(i))] for i in svm.predict(kernel_test)]
    #predictions = [test_dict[str(int(i))] for i in svm.predict(kernel_test)]
    print("Test images classified.")

    print("True labels:", len(true), true)
    print("Predictions:", len(predictions), predictions)


    #plotConfusions(true, predictions)
    #print("Confusion matrices plotted.")

    #findAccuracy(true, predictions)
    print("Measure performance -----------------------")
    print("Accuracy score %.3f" % metric.accuracy_score(true, predictions))
    print("F1-score %.3f" % metric.f1_score(true, predictions, average='macro'))
    print("Precision %.3f" % metric.precision_score(true, predictions, average='samples'))
    print("Recall %.3f" % metric.recall_score(true, predictions, average='samples'))
    #print("Accuracy calculated.")
    #print("Execution done.")

    time2 = time.time()
    print("Running testing time:", (time2 - time1) / 60, "min")




if __name__ == '__main__':

    imageProcessing = False
    if imageProcessing is True:
        img_path_train, annotation_path_train = getFilePath('dataset/Optisk')
        print("GetFilePath completed")

        images_train = readImageInFolder(img_path_train)
        print("readImageInFolder completed")

        scaled_images_train = scaleImageFromROI(images_train, annotation_path_train)
        print("scaleImageFromROI completed")

    # Split dataset into training and testing
    train_dict, test_dict = splitDataset()
    print("Train/test split completed")


    #print("Training images: ", len(train_images))
    #print("Testing images: ", len(test_images))


    train_model = False
    if train_model is True:
        kmeans, scale, svm, im_features = trainModel(train_dict, 10, 'linear')
        #kmeans, scale, svm, im_features = trainModel('Dataset/train', 10, 'linear')
        #trainModel(train_dict, 10, 'linear')

        # Save training variables
        myvar = [kmeans, scale, svm, im_features]
        with open('file.pkl', 'wb') as file:
            # A new file will be created
            pickle.dump(myvar, file)

        print("Pickle file saved")



    # Open saved file
    with open('file.pkl', 'rb') as file:
        # Call method
        var = pickle.load(file)

        #print(var[4])

    kmeans = var[0]
    scale = var[1]
    svm = var[2]
    im_features = var[3]

    print("Pickle file loaded")

    print(" ")
    #testModel('Dataset/test', kmeans, scale, svm, im_features, 10, 'linear')
    testModel(test_dict, kmeans, scale, svm, im_features, 10, 'linear')



