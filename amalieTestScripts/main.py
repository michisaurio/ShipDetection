"""
Implementation taken from:
https://medium.com/@aybukeyalcinerr/bag-of-visual-words-bovw-db9500331b2f
"""

import numpy as np
import cv2
import os                               # Allow you to interact with operating system. Standard Python utility modules
from scipy.spatial import distance
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from xml.etree import ElementTree as et


def getFilePath(folder):
    """
    Hent ut (returner) path til alle bildefilene og .xml filene
    :param folder:
    :return: array with filepaths: "dataset/Optisk/190616_112733.jpg/.xml"
    """
    file_path = []
    directory = os.listdir(folder)
    for filename in directory:
        path = folder + "/" + filename  # "dataset/Optisk/190616_112733.jpg"
        file_path.append(path)  # Legg til alle biler dataset/train/img_name.jpg

    # array[start_index:end_index:step]
    image_path = file_path[:len(file_path)-1:2]     # Exclude summary.xml file
    annotation_path = file_path[1::2]
    #print(len(image_path))

    #if (train is True):
    #    # Shuffle the image (paths) for training
    #    np.random.shuffle(image_path)
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
        img = cv2.imread(path, 0)   # Read image and convert to grayscale
        #img_resize = cv2.resize(img, (400, 300))
        #images.append(img_resize)
        images.append(img)
    return images


def changeLabelFilesInFolder(label_path):
    """
    Rescale the bounding boxes to fit the resized image
    :param label_path: The path of the annotated file (.xml)
    :return: None. Makes changes to the .xml file directly
    """

    # 'dataset/Optisk/190616_113607.xml'
    for path in label_path:
        print(path)

        tree = et.parse(path)
        myroot = tree.getroot()

        tag_name = ['xmin', 'ymin', 'xmax', 'ymax']

        for x in myroot.findall('object'):
            # Resize all bounding boxes
            print('Bounding box: ' + str(x))
            xmin = x.find('bndbox/' + tag_name[0])
            xmin.text = str(float(xmin.text)/10)
            ymin = x.find('bndbox/' + tag_name[1])
            ymin.text = str(float(ymin.text)/10)
            xmax = x.find('bndbox/' + tag_name[2])
            xmax.text = str(float(xmax.text)/10)
            ymax = x.find('bndbox/' + tag_name[3])
            ymax.text = str(float(ymax.text)/10)
    #    print(xmin.text, ymin.text, xmax.text, ymax.text)

        # Save to .xml file
        # TODO create a new .xml file to not override the original
        tree.write(path)


def cropImagesFromROI(images, label_path):
    """

    :param label_path:
    :return: bounding_box_vector
    """
    cropped_images = []
    print(len(label_path))
    for i in range(len(images)):
        tree = et.parse(label_path[i])
        myroot = tree.getroot()

        tag_name = ['xmin', 'ymin', 'xmax', 'ymax']

        tag = 0
        for x in myroot.findall('object'):
            name = x.find('name')

            if "motorboat" in name.text:
                print("In motorboat")
                xmin = x.find('bndbox/' + tag_name[0])
                print(xmin.text)
                ymin = x.find('bndbox/' + tag_name[1])
                print(ymin.text)
                xmax = x.find('bndbox/' + tag_name[2])
                print(xmax.text)
                ymax = x.find('bndbox/' + tag_name[3])
                print(ymax.text)

                tag += 1
                print(tag)

                img = images[i]
                ROI = img[int(float(ymin.text)):int(float(ymax.text)), int(float(xmin.text)):int(float(xmax.text))]
                # Save image to "result"-folder
                dir = 'dataset/Cropped'
                cv2.imwrite(os.path.join(dir, "cropped_img" + str(i) + "_" + str(tag) + ".jpg"), ROI)
                cv2.waitKey(0)

                cropped_images.append(ROI)


            if "sailboat" in name.text:
                print("In sailboat")
                xmin = x.find('bndbox/' + tag_name[0])
                print(xmin.text)
                ymin = x.find('bndbox/' + tag_name[1])
                print(ymin.text)
                xmax = x.find('bndbox/' + tag_name[2])
                print(xmax.text)
                ymax = x.find('bndbox/' + tag_name[3])
                print(ymax.text)

                tag += 1
                print(tag)

                img = images[i]
                ROI = img[int(float(ymin.text)):int(float(ymax.text)), int(float(xmin.text)):int(float(xmax.text))]
                # Save image to "result"-folder
                dir = 'dataset/Cropped'
                cv2.imwrite(os.path.join(dir, "cropped_img" + str(i) + "_" + str(tag) + ".jpg"), ROI)
                cv2.waitKey(0)

                cropped_images.append(ROI)

            if "barge" in name.text:
                print("In barge")
                xmin = x.find('bndbox/' + tag_name[0])
                print(xmin.text)
                ymin = x.find('bndbox/' + tag_name[1])
                print(ymin.text)
                xmax = x.find('bndbox/' + tag_name[2])
                print(xmax.text)
                ymax = x.find('bndbox/' + tag_name[3])
                print(ymax.text)

                tag += 1
                print(tag)

                img = images[i]
                ROI = img[int(float(ymin.text)):int(float(ymax.text)), int(float(xmin.text)):int(float(xmax.text))]
                # Save image to "result"-folder
                dir = 'dataset/Cropped'
                cv2.imwrite(os.path.join(dir, "cropped_img" + str(i) + "_" + str(tag) + ".jpg"), ROI)
                cv2.waitKey(0)

                cropped_images.append(ROI)

    return cropped_images

def SIFT_features(train_image):
    """
    Compute keypoints and descriptors
    :param train_image:
    :return: a list of descriptors
    """
    descriptor_list = []
    keypoint_img = []
    #sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=4, contrastThreshold=0.05, edgeThreshold=10, sigma=1.6)
    sift = cv2.SIFT_create()
    for img in train_image:
        # The value of the mask (None) can be provided when we are looking for the keypoints or features for a specific portion
        kp, des = sift.detectAndCompute(img, None)
        kp_on_img = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #kp_on_img = cv2.drawKeypoints(img, kp, None)
        descriptor_list.extend(des)
        keypoint_img.append(kp_on_img)

    return descriptor_list, keypoint_img


def clusterDescriptors(descriptors, k):
    """
    Cluster the list of descriptors using K-means clustering algorithm
    :param descriptors:
    :param k:
    :return: array of central points (AKA visual words)
    """
    kmeans = KMeans(n_clusters=k).fit(descriptors)
    visual_words = kmeans.cluster_centers_
    return visual_words


def vstackDescriptors(descriptor_list):
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))
    return descriptors


def find_index(image, center):
    count = 0
    ind = 0
    for i in range(len(center)):
        if(i == 0):
           count = distance.euclidean(image, center[i])
           #count = L1_dist(image, center[i])
        else:
            dist = distance.euclidean(image, center[i])
            #dist = L1_dist(image, center[i])
            if(dist < count):
                ind = i
                count = dist
    return ind


def image_class(all_bovw, centers):
    dict_feature = {}
    for key, value in all_bovw.items():
        category = []
        for img in value:
            histogram = np.zeros(len(centers))
            for each_feature in img:
                ind = find_index(each_feature, centers)
                histogram[ind] += 1
            category.append(histogram)
        dict_feature[key] = category
    return dict_feature


def showImage(images, kp_img):
    # Test image (190616_113607.jpg/.xml) without and with keypoints
    BRG_img = images[8]
    img = cv2.cvtColor(BRG_img, cv2.COLOR_BGR2RGB)
    img_kp = kp_img[8]

    # Display
    fig = plt.figure(figsize=(18,7))
    rows = 1
    columns = 2

    fig.add_subplot(rows, columns, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Original")

    fig.add_subplot(rows, columns, 2)
    plt.imshow(img_kp, cmap = plt.cm.gray)
    plt.axis('off')
    plt.title("Image with keypoints")
    plt.show()





#def trainModel(train_path):
#    images = getFiles(train_path)
#    return images


if __name__ == '__main__':
    # TODO create train_model() and test_model() methods to run in _main_
    # TODO lag en funksjon som splitter datasettet i training,testing (trainingSet, testingSet = splitDataset(img))
    # Training
    #image_path_train = getFiles(True, 'dataset/train')
    image_path_train, annotation_path_train = getFilePath('dataset/Optisk') # TODO legg til True for shuffeling training images
    image_count = len(image_path_train)

    train_images = readImageInFolder(image_path_train)

    """
    change_annotation_file = False
    if change_annotation_file is True:
        # If True, make changes to the .xml files in 'dataset/Optisk'
        # Run only once
        changeLabelFilesInFolder(annotation_path_train)
        print("NOTE: Bounding boxes are resized")
    """

    img_crop = cropImagesFromROI(train_images, annotation_path_train)


    # Compute descriptors from images
    descriptor_list, keypoint_img = SIFT_features(img_crop)
    print("Descriptors computed")

    # Show result on image
    #showImage(img_crop, keypoint_img)

    descriptors = vstackDescriptors(descriptor_list) # Er ikke helt sikker på hvorfor dette gjøres
    print("Descriptors vstacked.")

    # Create visual vocabulary
    # Send the visual dictionary to the k-means clustering algorithm
    #visual_words = clusterDescriptors(descriptors, 10)  # Takes the central points which is visual words
    #print("Descriptors clustered")

    # Creates histograms for train data
    #bovw_train = image_class(descriptor_list, visual_words)

    """
    # Testing
    #image_path_test = getFiles(False, 'dataset/test')
    #test_images = readImage(image_path_test)
    print("Reading images completed")

    #dir = 'dataset/result'
    #cv2.imwrite(os.path.join(dir, "result.jpg"), train_images[0])
    #cv2.waitKey(0)

    descriptor_list = SIFT_features(train_images)
    print("Descriptors computed")

    visual_words = clusterDescriptors(descriptor_list, k=50)
    print(visual_words[0])
    print("Clustering completed")

    


    img_BRG = cv2.imread('dataset/ship/ship1.jpg')
    img = cv2.cvtColor(img_BRG, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img, (400, 300))

    # Compare different SIFT objects
    #compare_SIFT(img_resize)

    blur1 = cv2.blur(img_resize, (5, 5)) # Gives same result as with kernel
    blur2 = cv2.blur(img_resize, (7, 7))
    compute_and_show_keypoints(blur1, blur2, 'Blur, 5x5 kernel', 'Blur, 7x7 kernel')

    median1 = cv2.medianBlur(img_resize, 5) # Kernel size must be positive, odd integer
    median2 = cv2.medianBlur(img_resize, 7)
    compute_and_show_keypoints(median1, median2, 'Median, 5x5 kernel', 'Median, 7x7 kernel')

    bilateral1 = cv2.bilateralFilter(img_resize, 2, 75, 75)
    bilateral2 = cv2.bilateralFilter(img_resize, 10, 75, 75)
    compute_and_show_keypoints(bilateral1, bilateral2, 'Bilateral less', 'Bilateral more')
 
    titles = ['Original Image', 'Blurred', 'Median', 'Bilateral']
    images = [kp_on_img4, kp_on_img1, kp_on_img2, kp_on_img3]

    fig = plt.figure(figsize=(15, 7))
    for i in range(4):
        fig.add_subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
       """
