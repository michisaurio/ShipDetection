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


"""
Hent ut (returner) path til alle bildefilene og .xml filene
Return array med "dataset/Optisk/190616_112733.jpg" osv
Return array med "dataset/Optisk/190616_112733.xml" osv
"""
def getFilePath(folder):
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
        img_resize = cv2.resize(img, (400, 300))
        images.append(img_resize)
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
        kp_on_img = cv2.drawKeypoints(img, kp, None)
        descriptor_list.extend(des)
        keypoint_img.append(kp_on_img)

    return descriptor_list, keypoint_img


def extractFeatures(visual_words, descriptor_list, image_count, no_clusters):
    image_features = np.array([np.zeros(no_clusters) for i in range(image_count)])
    for i in range(image_count): # 0 - antall bilder
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
    #        feature = feature.reshape(1, 128)
            idx = visual_words.predict(feature)
            image_features[i][idx] += 1
    return image_features


"""
Cluster the list of descriptors using K-means clustering algorithm
Return array of central points (AKA visual words)
"""
def clusterDescriptors(descriptors, k):
    kmeans = KMeans(n_clusters=k).fit(descriptors)
    visual_words = kmeans.cluster_centers_
    return visual_words


#def trainModel(train_path):
#    images = getFiles(train_path)
#    return images


if __name__ == '__main__':
    # TODO create train_model() and test_model() methods to run in _main_
    # Training
    #image_path_train = getFiles(True, 'dataset/train')
    image_path_train, annotation_path_train = getFilePath('dataset/Optisk') # TODO legg til True for shuffeling training images

    train_images = readImageInFolder(image_path_train)

    change_annotation_file = False
    if change_annotation_file is True:
        # If True, make changes to the .xml files in 'dataset/Optisk'
        # Run only 1 time
        changeLabelFilesInFolder(annotation_path_train)
        print("NOTE: Bounding boxes are resized")


    # Compute descriptors from images
    descriptor_list, keypoint_img = SIFT_features(train_images)

    # Test image (190616_113607.jpg/.xml) without and with keypoints
    img = train_images[14]
    img_kp = keypoint_img[14]

    dir = 'result'
    cv2.imwrite(os.path.join(dir, "test_image.jpg"), img)
    cv2.imwrite(os.path.join(dir, "test_image_kp.jpg"), img_kp)   # Check keypoints on image
    cv2.waitKey(0)


    #image_count = len(image_path_train)



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

    im_features = extractFeatures(visual_words, descriptor_list, image_count, 50)
    print("Images features extracted")


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
