import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metric
import time
from sklearn.metrics import classification_report
from imageProcessing import ImageProcessing
import pickle


def getClassImages(path):
    """
    Get images from the folder in 'path'
    :param path: The path to the folder containing the dataset files
    :return class_images: Images in the folder
    """
    class_images = []
    file_path = []
    for filename in os.listdir(path):
        n_path = path + "/" + filename  # "dataset/Optisk/190616_112733.jpg"
        file_path.append(n_path)
        img = cv2.imread(n_path)  # Read image
        class_images.append(img)
    return class_images


def splitDataset():
    """
     Split dataset into training and testingset
     :return train_dict, test_dict: Dictionaries containing training and testing images
     """
    # Get the classified images
    motorboat_priority = getClassImages('Cropped_bilateral/motorboat_priority')
    sailboat_down = getClassImages('Cropped_bilateral/sailboat_down')
    sailboat_up = getClassImages('Cropped_bilateral/sailboat_up')
    barge = getClassImages('Cropped_bilateral/barge')
    motorboat = getClassImages('Cropped_bilateral/motorboat')

    dataset = [motorboat_priority, sailboat_down, sailboat_up, barge, motorboat]

    train = []
    test = []
    for i in dataset:
        train_i, test_i = train_test_split(i, test_size=0.2, train_size=0.8, shuffle=True)
        train.append(train_i)
        test.append(test_i)

    train_dict = {
        "motorboat_priority": train[0],  # motorboat_priority
        "sailboat_down": train[1],  # sailboat_down
        "sailboat_up": train[2],  # sailboat_up
        "barge": train[3],  # barge
        "motorboat": train[4]  # motorboat
    }

    test_dict = {
        "motorboat_priority": test[0],  # motorboat_priority
        "sailboat_down": test[1],  # sailboat_down
        "sailboat_up": test[2],  # sailboat_up
        "barge": test[3],  # barge
        "motorboat": test[4]  # motorboat
    }
    return train_dict, test_dict


def getDescriptors(sift, orb, extractor, img):
    """
    Compute keypoints and descriptors
    :param sift: SIFT feature extractor
    :param orb: ORB feature extractor
    :param extractor: String containing "orb" or "sift"
    :param img: Image to execute feature extraction on
    :return des, img_kp: Array with descriptors and image containing keypoints
    """
    if "sift" in extractor:
        kp, des = sift.detectAndCompute(img, None)

    if "orb" in extractor:
        kp, des = orb.detectAndCompute(img, None)

    img_kp = cv2.drawKeypoints(img, kp, None, color=(0, 0, 255))
    return des, img_kp


def vstackDescriptors(descriptor_list):
    """
    Vertically stack descriptors in the array 'descriptor_list'
    :param descriptor_list: The array containing the descriptors
    :return descriptors: Vertically stacked array
    """
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))
    return descriptors


def clusterDescriptors(descriptors, no_clusters):
    """
    Cluster descriptors using k-means cluster algorithm
    :param descriptors: Array of descriptors
    :param no_clusters: Number of clusters
    :return kmeans:
    """
    kmeans = KMeans(n_clusters=no_clusters, init='k-means++').fit(descriptors)
    # The descriptors are now divided into no_clusters clusters
    return kmeans


def extractFeatures(kmeans, descriptor_list, extractor, image_count, no_clusters):
    im_features = np.array([np.zeros(no_clusters) for i in range(image_count)])
    for i in range(image_count):
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
            if "sift" in extractor:
                feature = feature.reshape(1, 128)
            else:
                feature = feature.reshape(1, 32)
            idx = kmeans.predict(feature)
            im_features[i][idx] += 1
    return im_features


def normalizeFeatures(scale, features):
    return scale.transform(features)


def plotHistogram(im_features, no_clusters, train):
    x_scalar = np.arange(no_clusters)
    y_scalar = np.array([abs(np.sum(im_features[:, h], dtype=np.int32)) for h in range(no_clusters)])

    plt.bar(x_scalar, y_scalar)
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    if train is True:
        plt.title("Complete Train Vocabulary Generated")
    else:
        plt.title("Complete Test Vocabulary Generated")
    plt.xticks(x_scalar + 0.4, x_scalar)
    plt.show()


def svcParamSelection(X, y, kernel, nfolds):
    """
    Find the best suitable parameter values using grid search
    :param X:
    :param y:
    :param kernel:
    :param nfolds: Numbers of cross-validation folds
    :return: The best parameter value
    """
    C = [0.001, 0.01, 0.1, 1, 10]
    gamma = [0.01, 0.1, 1, 10]
    param_grid = {'C': C, 'gamma': gamma}
    grid_search = GridSearchCV(SVC(kernel=kernel), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    print("Best parameter values found on development set:")
    print(grid_search.best_params_)
    print("")
    return grid_search.best_params_


def findSVM(im_features, train_labels, kernel):
    """
    Classify the predicted labels using SVM
    :param im_features:
    :param train_labels:
    :param kernel:
    :return:
    """
    features = im_features

    params = svcParamSelection(features, train_labels, kernel, 5)
    C_param, gamma_param = params.get("C"), params.get("gamma")
    print(C_param, gamma_param)
    class_weight = {
        0: (1565 / (5 * 826)),  # motorboat_priority
        1: (1565 / (5 * 79)),  # sailboat_down
        2: (1565 / (5 * 24)),  # sailboat_up
        3: (1565 / (5 * 24)),  # barge
        4: (1565 / (5 * 612))  # motorboat
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

    class_names = ["barge", "motorboat", "motorboat w/ priority", "sailboat w/ sail down", "sailboat w/ sail up"]
    plotConfusionMatrix(true, predictions, classes=class_names,
                        title='Confusion matrix, without normalization')

    plotConfusionMatrix(true, predictions, classes=class_names, normalize=True,
                        title='Normalized confusion matrix')
    plt.show()


def findAccuracy(true, predictions):
    """
    Print out performance metrics
    :param true: True labels
    :param predictions: Predicted labels
    """
    print("Accuracy score %.3f" % metric.accuracy_score(true, predictions))
    print("F1-score (macro) %.3f" % metric.f1_score(true, predictions, average='macro'))
    print("F1-score (micro) %.3f" % metric.f1_score(true, predictions, average='micro'))

    # Classification report
    class_names = ["barge", "motorboat", "motorboat w/ priority", "sailboat w/ sail down", "sailboat w/ sail up"]
    print(classification_report(true, predictions, target_names=class_names, sample_weight=None, digits=2,
                                output_dict=False, zero_division='warn'))


def plotCodewords(codewords, nr_clusters, rows, columns):
    # Visualizing codewords/visual words
    for i in range(nr_clusters):
        plt.subplot(rows, columns, i + 1), plt.imshow(codewords[i].reshape(16, 8))
        plt.xticks([]), plt.yticks([])
    plt.show()


def trainModel(train_dict, extractor, no_clusters, kernel):
    time1 = time.time()  # Start timer

    # Create extractors
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0,
                         WTA_K=2, patchSize=31, fastThreshold=10)

    descriptor_list = []
    kp_img_list = []
    train_labels = np.array([])
    count = 0

    train_images = []
    for k, img_list in train_dict.items():
        for img in img_list:
            des, kp_img = getDescriptors(sift, orb, extractor, img)
            train_images.append(img)

            if des is not None:
                count += 1
                descriptor_list.append(des)
                kp_img_list.append(kp_img)
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

    print("Step 1/7 : Descriptors computed. ")

    print("Number of images:", len(train_images))
    print("Number of train labels:", len(train_labels))
    print("Number of images removed:", len(train_images) - len(train_labels))

    print("Descriptor list is of shape", np.shape(descriptor_list))
    print('Descriptors are vectors of shape', descriptor_list[0].shape)
    print("")
    # dir = 'images'
    # cv2.imwrite(os.path.join(dir, "descriptor_0_ORB.jpg"), descriptor_list[0])
    # cv2.waitKey(0)

    descriptors = vstackDescriptors(descriptor_list)
    print("Descriptor list vstacked has shape:", np.shape(descriptors))
    print("Descriptors are vectors of shape:", np.shape(descriptors[0]))
    print("Step 2/7: Descriptors vstacked.")

    kmeans = clusterDescriptors(descriptors, no_clusters)
    codewords = kmeans.cluster_centers_
    plotCodewords(codewords, no_clusters, rows=2, columns=5)
    print("Step 3/7: Descriptors clustered.")

    im_features = extractFeatures(kmeans, descriptor_list, extractor, count, no_clusters)
    print("Image features of shape:", np.shape(im_features))
    print("Step 4/7: Images features extracted.")
    print("")

    scale = StandardScaler().fit(im_features)
    im_features_scale = scale.transform(im_features)
    print("Step 5/7: Train images normalized.")

    plotHistogram(im_features_scale, no_clusters, True)
    print("Step 6/7: Features histogram plotted.")

    svm = findSVM(im_features_scale, train_labels, kernel)
    print("Get SVM parameters: ")
    print(svm.get_params())
    print("Step 7/7 : SVM fitted.")

    print("")
    print("Training completed.")

    time2 = time.time()
    print("... Running training time:", (time2 - time1) / 60, "min ...")

    return kmeans, scale, svm


def testModel(test_dict, extractor, kmeans, scale, svm, no_clusters):
    time1 = time.time()

    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0,
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

    for k, img_list in test_dict.items():
        for img in img_list:
            des, _ = getDescriptors(sift, orb, extractor, img)

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
    print("Step 1/5: Descriptors computed.")
    print("")

    test_features = extractFeatures(kmeans, descriptor_list, extractor, count, no_clusters)

    test_features = scale.transform(test_features)
    print("Nr. of test features:", len(test_features))

    plotHistogram(test_features, no_clusters, False)
    print("Step 2/5: Features histogram plotted.")

    kernel_test = test_features

    predictions = [name_dict[str(int(i))] for i in svm.predict(kernel_test)]
    print("Step 3/5: Test images classified.")

    plotConfusions(true, predictions)
    print("Step 4/5: Confusion matrices plotted.")
    print("")

    print("Measure performance -----------------------")
    findAccuracy(true, predictions)
    print("Step 5/5: Performance calculated.")
    print("Execution done.")

    time2 = time.time()
    print("Running testing time:", (time2 - time1) / 60, "min")


if __name__ == '__main__':

    var = False     # Set to True to pre-process the dataset (cropped_images)
    if var is True:
        image_processing = ImageProcessing()
        img_path_train, annotation_path_train = image_processing.getFilePath('dataset/Optisk')
        print("GetFilePath completed")
        images_train = image_processing.readImageInFolder(img_path_train, False)
        print("readImageInFolder completed")
        scaled_images_train = image_processing.cropImageFromROI(images_train, annotation_path_train)
        print("scaleImageFromROI completed")


    train_dict, test_dict = splitDataset()
    print("Train/test split completed")

    extractor = "sift"
    noClusters = 10

    kmeans, scale, svm = trainModel(train_dict, extractor=extractor, no_clusters=noClusters, kernel='rbf')
    print(" ")
    testModel(test_dict, extractor, kmeans, scale, svm, no_clusters=noClusters)

    """
        # Save training variables
    myvar = [kmeans, img_features]
    with open('file.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(myvar, file)
    print("Pickle file saved")
    
    # Open saved file
    with open('file.pkl', 'rb') as file:
        # Call method
        var = pickle.load(file)

    kmeans = var[0]
    img_features = var[1]

    print("Pickle file loaded")
    """











