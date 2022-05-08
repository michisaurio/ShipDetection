import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas
from scipy import ndimage
from scipy.spatial import distance
from skimage.transform import resize
import os
from skimage.feature import hog
from skimage import exposure
import math
import random
from random import randint
from skimage import feature
from imageProcessing import ImageProcessing
import time
from sklearn.cluster import KMeans
from scipy import spatial
import pickle
import seaborn as sns
from sklearn.decomposition import PCA


def sliding_window(image, stepSizeX, stepSizeY, windowSize):
    """
    Sliding window function
    :param image: The image we are going to loop/slide over
    :param stepSize: How many pixels we are going to "skip" in both the (x, y) direction.
    :param windowSize: Width and height (in terms of pixels) of the window we are going to extract from out image
    :return:
    """
    # Slide a window across the image
    for y in range(0, image.shape[0], stepSizeY):
        for x in range(0, image.shape[1], stepSizeX):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def downsampleImages(images):
    sampled_images = []
    for img in images:
        # Use lowpass filter on image - Blur
        # blur = cv2.blur(img, (5, 5))
        # Downsample image
        img_shape = np.shape(img)
        img_resize_2 = cv2.resize(img, (int(img_shape[1] / 2), int(img_shape[0] / 2)))  # factor 2
        img_resize_4 = cv2.resize(img, (int(img_shape[1] / 4), int(img_shape[0] / 4)))  # factor 4

        # Add all images to dataset
        sampled_images.append(img)
        sampled_images.append(img_resize_2)
        sampled_images.append(img_resize_4)
    # print("Image size after downsampling:", np.shape(img_resize_2), np.shape(img_resize_4))
    return sampled_images


def getFilePath(folder):
    file_path = []
    directory = os.listdir(folder)
    for filename in directory:
        path = folder + "/" + filename  # "dataset/Optisk/190616_112733.jpg"
        file_path.append(path)
    return file_path


def readImageInFolder(path):
    images = []
    for img_path in path:
        img = cv2.imread(img_path)  # Read image and convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = np.float32(gray)
        # images.append(img)
        images.append(gray)
    return images


def showOneImage(image, map, title):
    plt.imshow(image, cmap=map)
    plt.title(title)
    plt.show()


def saveImageToFolder(image, img_name, dir):
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(dir, img_name), image)
    cv2.waitKey(0)


def getPatches(train_images, info, save, save_dir):
    image_patches = []
    variance = []
    tag = 0
    # stride = int(window_width * 0.25)

    for input_img in train_images:
        tag += 1
        img_shape = np.shape(input_img)  # (0:height, 1:width)

        window_width = 64  # int(img_shape[1] / 6)
        window_height = 64  # int(img_shape[0] / 6)
        stride_x = int(window_width * 0.5)  # int(window_width * 0.25)
        stride_y = int(window_height * 0.5)

        # Create output image to store patches
        output_img = np.full(shape=(img_shape[0], img_shape[1]), fill_value=255, dtype=np.int)
        # print("Shape of output image", str(tag), ":", np.shape(output_img))

        # Compute Harris response
        # dst_map = cv2.cornerHarris(input_img, blockSize=3, ksize=3, k=0.06)

        # Sliding window algorithm
        for (x, y, window) in sliding_window(input_img, stepSizeX=stride_x, stepSizeY=stride_y,
                                             windowSize=(window_width, window_height)):
            # if our window does not meet our desired window size, ignore it
            if window.shape[0] != window_height or window.shape[1] != window_width:
                continue

            # THIS IS THE PLACE TO PROCESS YOUR WINDOW. Add descriptors to describe the content of the window (?)

            # Draw the window
            clone = input_img.copy()
            cv2.rectangle(clone, (x, y), (x + window_width, y + window_height), (0, 255, 0), 4)
            # rectangle(image to draw, start coordinate, end coordinates, color, thickness)

            # window_patch_dst = dst_map[y:y + window_height, x:x + window_width]
            window_patch_gray = input_img[y:y + window_height, x:x + window_width]
            # window_patch_gray = gray[y:y + window_height, x:x + window_width]  # For visualizing image patches
            # dst = cv2.cornerHarris(window_patch, blockSize=3, ksize=3, k=0.06)

            # TODO finn ut om dette er riktig måte å gjøre det på. Spør E
            # Compute variance
            intensity = ndimage.variance(window_patch_gray)
            # intensity = np.sum(np.abs(window_patch_dst))
            variance.append(intensity)

            # Print info
            if info is True:
                # print("")
                print("Image patch", tag)
                print("Start coordinates:", (x, y))
                print("End coordinates:", (x + window_width, y + window_height))
                print("Intensity:", intensity, '\n')

            # Show sliding window + corner response image patch
            # showOneImage(image=clone, map=map, title="Grayscale image with sliding window")
            # showOneImage(image=window_patch_gray, map=map, title="Patch from sliding window")

            # plt.subplot(121)
            # plt.imshow(window_patch, cmap=map)
            # plt.title("Image patch")
            # plt.subplot(122)
            # plt.imshow(dst_map, cmap=map)
            # plt.title("Harris response")
            # plt.show()

            # Intensity threshold
            var_threshold = 1500
            if intensity > var_threshold:
                # Store image patches
                image_patches.append(window_patch_gray)
                # Add image patch to output_img on the same position it was taken out
                output_img[y:y + window_height, x:x + window_width] = window_patch_gray
                if save is True:
                    img_name = "image_" + str(tag) + ".jpg"
                    saveImageToFolder(output_img, img_name, save_dir)
                # print("Image patch stored!")
    return image_patches, variance


def runSingleImage(filename):
    # TODO Read images inside bounding box of ship
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = np.float32(gray)

    img_shape = np.shape(gray)  # (0:height, 1:width)
    print("Size of image:", img_shape, "\n")

    map = 'gray'

    # Create output image
    output_img = np.full(shape=(img_shape[0], img_shape[1]), fill_value=255, dtype=np.int)
    print("Shape of output image:", np.shape(output_img))

    # Compute Harris response map
    dst_map = cv2.cornerHarris(gray, blockSize=3, ksize=3, k=0.06)
    # plt.imshow(dst_map, cmap=map)
    # plt.title("Harris response map")
    # plt.show()

    # img_name = 'harris_response.jpg'
    # cv2.imwrite(os.path.join('images/image_patches', img_name), dst_map)
    # cv2.waitKey(0)

    # HarrisCornerOpenCV(img, gray, map)

    # Downsample image
    img_resize_2 = cv2.resize(gray, (int(img_shape[1] / 2), int(img_shape[0] / 2)))  # factor 2
    img_resize_4 = cv2.resize(gray, (int(img_shape[1] / 4), int(img_shape[0] / 4)))  # factor 4
    print("Image size after downsampling:", np.shape(img_resize_2), np.shape(img_resize_4))
    # plt.imshow(img_resize_4, cmap=map)
    # plt.show()

    (window_width, window_height) = (128, 128)

    input_img = dst_map

    image_patches = []
    var_arr = []
    tag = 0
    stride = int(window_width * 0.25)

    # Use sliding window over one image - input_img
    for (x, y, window) in sliding_window(input_img, stepSizeX=stride, stepSizeY=stride,
                                         windowSize=(window_width, window_height)):
        # if our window does not meet our desired window size, ignore it
        if window.shape[0] != window_height or window.shape[1] != window_width:
            continue

        # THIS IS THE PLACE TO PROCESS YOUR WINDOW. Add descriptors to describe the content of the window (?)

        # Draw the window
        clone = input_img.copy()
        cv2.rectangle(clone, (x, y), (x + window_width, y + window_height), (0, 255, 0), 4)

        # Compute corner response map
        window_patch = input_img[y:y + window_height, x:x + window_width]  # For computing intensity
        window_patch_gray = gray[y:y + window_height, x:x + window_width]  # For visualizing image patches

        # Compute variance
        intensity = np.sum(np.abs(window_patch))
        var_arr.append(intensity)

        # Print info
        info = False
        if info is True:
            tag += 1
            # print("")
            print("Image patch", tag)
            print("Start coordinates:", (x, y))
            print("End coordinates:", (x + window_width, y + window_height))
            print("Intensity:", intensity, '\n')

        # Show sliding window + corner response image patch
        # plt.imshow(clone, cmap=map)
        # plt.title("Grayscale image with sliding window")
        # plt.show()

        # plt.imshow(window_patch, cmap=map)
        # plt.title("Patch from sliding window")
        # plt.show()

        # plt.subplot(121)
        # plt.imshow(window_patch, cmap=map)
        # plt.title("Image patch")
        # plt.subplot(122)
        # plt.imshow(dst_map, cmap=map)
        # plt.title("Harris response")
        # plt.show()

        # Intensity threshold
        save = False
        dir = 'images/image_patches'

        var_threshold = 1e9
        if intensity > var_threshold:
            # Store image patches
            image_patches.append(window_patch_gray)
            # Add image patch to output_img on the same position it was taken out
            output_img[y:y + window_height, x:x + window_width] = window_patch_gray
            # print("Image patch added!")
            if save is True:
                img_name = "patch" + "_" + str(tag) + ".jpg"
                cv2.imwrite(os.path.join(dir, img_name), window_patch_gray)
                cv2.waitKey(0)
            # print("Image patch stored!")

        # Store image patches
        # image_patch = gray[y:y + window_height, x:x + window_width]
        # image_patches.append(image_patch)
        # plt.imshow(image_patch)
        # plt.show()

    print("")
    print("Number of image patches stored:", len(image_patches))

    # Add window on output_img (For reference)
    out_clone = output_img.copy()
    cv2.rectangle(out_clone, (0, 0), (0 + window_width, 0 + window_height), (0, 255, 0), 4)
    # rectangle(image to draw, start coordinate, end coordinates, color, thickness)
    # Show output_img
    plt.subplot(121)
    plt.imshow(gray, cmap=map)
    plt.title("Original image")
    plt.subplot(122)
    plt.imshow(out_clone, cmap=map)
    plt.title("")
    plt.show()

    # Histogram to determine variance threshold
    # plt.hist(var_arr, bins=[0, 1e10, 2e10, 3e10, 4e10, 5e10, 6e10, 7e10, 8e10], rwidth=0.9)

    # plt.hist(var_arr, rwidth=0.9)  # , bins=[15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], rwidth=0.9)
    # plt.grid()
    # plt.show()

    """
    # Show image patches
    #for i in range(len(image_patches)-2):
    #    plt.subplot(10, 9, i + 1), plt.imshow(image_patches[i], cmap=map)
    #    plt.xticks([]), plt.yticks([])
    #plt.show()

    #HarrisCornerAlgorithm(I=gray, alpha=0.05, T_response=20000, map=map)
    """


def elbowMethod(list_to_cluster, range):
    # elbow method
    # wcss = []
    # for i in range(1, 100):
    #    k_means = KMeans(n_clusters=i, init='k-means++', random_state=42)
    #    k_means.fit(list_to_cluster)
    #    wcss.append(k_means.inertia_)
    # plot elbow curve
    # plt.plot(np.arange(1, 100), wcss)
    # plt.xlabel('Clusters')
    # plt.ylabel('SSE')
    # plt.show()
    inertia = []
    K = range(1, range)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(list_to_cluster)
        kmeanModel.fit(list_to_cluster)
        inertia.append(kmeanModel.inertia_)
    # Plot the elbow
    plt.plot(K, inertia, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.show()


def printDescriptorIndexFromKey(key, descriptors, cluster_labels):
    # Lag dictionary med index til descriptor med tilhørende cluster label
    index_kmeans = {}
    for index in range(len(descriptors)):
        index_kmeans[index] = str(cluster_labels[index])
    # print(index_kmeans, '\n')

    index_des = {}
    for pair in index_kmeans.items():
        if pair[1] not in index_des.keys():
            index_des[pair[1]] = []
        index_des[pair[1]].append(pair[0])
    # print(index_des)
    # print("In the following cluster, the index of the descriptors are given as:")
    # print(index_des.get(key))
    return index_des.get(key)


def computeKmeans(n_dictionary, v, init, state, des):
    kmeans = KMeans(n_clusters=n_dictionary,
                    verbose=v,
                    init=init,
                    random_state=state,
                    n_init=3)
    # fit the model
    kmeans.fit(des)
    return kmeans


def getClusterCentres(kmeans):
    centroid = kmeans.cluster_centers_
    return centroid


def getVocabulary(no_clusters, descriptor_list, kmeans_labels, codeword, N):
    words = []
    for key in range(no_clusters):
        # for alle centroids (50 stk)
        index_des = printDescriptorIndexFromKey(str(key), descriptor_list, kmeans_labels)
        # print("Number of descriptors in cluster nr.", str(key), ":", len(index_des))

        cluster_descriptor = []
        for i in index_des:
            cluster_descriptor.append(descriptor_list[i])
            # Har nå en liste med descriptorer fra cluster [key]
        # computes distances for cluster 1
        distances = []
        for ind in range(len(index_des)):
            # Euclidean distance = cluster - centroid
            # euclidean = np.sqrt(np.sum((cluster_descriptor[ind] - codeword[key]) ** 2))  # compute distance between test image and any image in training set
            euclidean = distance.euclidean(codeword[key], descriptor_list[ind])
            distances.append(euclidean)
        # print("Shape of my_list:", np.shape(my_list), "\n")

        # Get largest element
        # maxElement = np.amax(my_list)
        # index = my_list.index(maxElement)
        # ind_descriptor = index_des[index]
        # print(maxElement, index, ind_descriptor)
        # words.append(ind_descriptor)
        words_2 = []
        # Get index for top 5 elements
        indx = np.argpartition(distances, N)[:N]
        # indx = np.argpartition(distances, -N)[-N:]
        for i in indx:
            ind_descriptor = index_des[i]
            words_2.append(ind_descriptor)
        # ind_descriptor2 = index_des[index]
        # print(index2, "\n")
        words.append(words_2)
    return words


if __name__ == '__main__':
    time1 = time.time()  # Start timer


    ship_file_path = getFilePath('images/cropped/A_ship')
    print("Number of ship file path:", len(ship_file_path), '\n')


    gray_ship_images = readImageInFolder(ship_file_path)
    print("Number of ship images:", len(gray_ship_images), '\n')



    # TODO downsample images (factor 2 and 4)
    #train_images = downsampleImages(gray_images)
    train_images_ship = gray_ship_images

    map = 'gray'

    # Visualize all images
    #for i in range(len(train_images)):
    #    plt.subplot(2, 5, i + 1), plt.imshow(train_images[i], cmap=map)
    #    plt.xticks([]), plt.yticks([])
    #plt.show()


    # Compute Harris corner response map for each image
    #dst_map = []
    #for img in train_images:
        # Compute Harris response map
    #    dst = cv2.cornerHarris(img, blockSize=3, ksize=3, k=0.06)
    #    dst_map.append(dst)
    #print("Number of harris response maps:", len(dst_map))

    # Store grayscale and harris in dictionary
    #dataset = {i:[] for i in range(1, 3)}
    #dataset[1] = train_images
    #dataset[2] = dst_map

    #values = list(dataset.values())
    #key = 0     # 0: grayscale images, 1: harris response map
    #img_nr = 1
    #plt.imshow(values[key][img_nr], cmap=map)
    #plt.show()

    # Get one image from train_images
    #for value in dataset.values():
    #    print(value[0]) # print image 0 for each key

    dir = 'images/image_patches2'
    image_patches_ship, var_arr_ship = getPatches(train_images_ship, info=False, save=False, save_dir=dir)

    print("Total number of extracted patches [ship]:", len(var_arr_ship))
    print("Number of image patches (with threshold) [ship]:", len(image_patches_ship), '\n')


    #name = 0
    #for patch in image_patches:
    #    name += 1
    #    img_name = str(name) + ".jpg"
    #    saveImageToFolder(patch, img_name, dir)

    # Check variances from each image patch
    plt.style.use('seaborn-deep')
    show_hist = False
    if show_hist is True:
        plt.hist(var_arr_ship, rwidth=0.9)  # , bins=[15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], rwidth=0.9)
        plt.grid()
        plt.title("Histogram of Variance")
        plt.xlabel("Values")
        plt.ylabel("Frequency")
        plt.axvline(x=1500, ymin=0, ymax=250000, color='red', linestyle='dashed', linewidth=1)
        plt.text(x=1520, y=150000, s='Threshold', fontsize=10)
        plt.show()

    # Compute descriptor for each image patch
    descriptors_ship = []
    for patch in image_patches_ship:
        #print("*** NEW PATCH ***")
        # Get coordinates of centre point
        shape_patch = np.shape(patch)
        length = shape_patch[1]
        width = shape_patch[0]
        #print("shape image patch:", shape_patch)

        #kp = (int(length/2), int(width/2)) # (x_c, y_c)
        #print("Keypoint coordinate:", kp)


        #plt.subplot(121)
        #plt.imshow(patch, cmap=map)
        #plt.subplot(122)
        #plt.imshow(resized_patch, cmap=map)
        #plt.show()


        # Compute HoG descriptor
        #resized_patch = resize(patch, (128, 64))
        des_hog, hog_image = hog(patch, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=False)
        #print("HoG features:", np.shape(des_hog), '\n')
        #print(des_hog)
        descriptors_ship.append(des_hog)
        #hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        #plt.imshow(hog_image_rescaled, cmap=map)
        #plt.show()


    print("Number of HOG features:", np.shape(descriptors_ship))

    save = False
    if save is True:
        # Save descriptors to file
        myvar = [descriptors_ship, image_patches_ship]
        with open('dataset_binary.pkl', 'wb') as file:
            # A new file will be created
            pickle.dump(myvar, file)
        print("Pickle file saved")
    else:
        # Open saved file
        with open('dataset_binary.pkl', 'rb') as file:
            # Call method
            var = pickle.load(file)

        descriptors_ship = var[0]
        descriptors_nonship = var[1]
        print("Pickle file loaded", '\n')

    # Compare distances between descriptors
    # i = 253
    # j = 158
    # euclidean = np.sqrt(np.sum((descriptors[i] - descriptors[j]) ** 2))
    # cosine_dist = 1 - spatial.distance.cosine(descriptors[i], descriptors[j])
    # print("Distance between descriptor", str(i), "and", str(j), ":", cosine_dist)

    no_clusters = 100  # size of the dictionary
    random_state = 1

    # elbowMethod(descriptors, range=10)

    # Principal Component Analysis for decreasing descriptor dimension
    # pca = PCA(2)
    # df = pca.fit_transform(descriptors)
    # print("PCA shape:", np.shape(df))

    # Define a KMeans clustering model
    kmeans_model_ship = computeKmeans(no_clusters, v=False, init='random', state=random_state, des=descriptors_ship)
    # labels = kmeans_model.labels_  # Label for each descriptor of the training set
    labels_ship = kmeans_model_ship.predict(descriptors_ship)
    print("Shape labels:", np.shape(labels_ship))

    # centroids_ship = kmeans_model_ship.cluster_centers_


    """
    # Pick a random number between 1 and no_cluster
    #label0 = df[labels == 0]
    #label1 = df[labels == 2]
    #label2 = df[labels == 20]
    #label3 = df[labels == 25]
    #label4 = df[labels == 40]
    #label5 = df[labels == 45]
    #label6 = df[labels == 60]
    #label7 = df[labels == 65]
    #label8 = df[labels == 80]
    #label9 = df[labels == 90]
    #print("Shape label0:", np.shape(label0))

    #plt.scatter(label0[:, 0], label0[:, 1], color='blue')
    #plt.scatter(label1[:, 0], label1[:, 1], color='yellow')
    #plt.scatter(label2[:, 0], label2[:, 1], color='pink')
    #plt.scatter(label3[:, 0], label3[:, 1], color='forestgreen')
    #plt.scatter(label4[:, 0], label4[:, 1], color='blueviolet')
    #plt.scatter(label5[:, 0], label5[:, 1], color='orange')
    #plt.scatter(label6[:, 0], label6[:, 1], color='peru')
    ##plt.scatter(label7[:, 0], label7[:, 1], color='teal')
    #plt.scatter(label8[:, 0], label8[:, 1], color='navy')
    #plt.scatter(label9[:, 0], label9[:, 1], color='cyan')
    #plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='red')
    #plt.show()

    # plot all k-means clusters
    # Plot all clusters
    u_labels = np.unique(labels)
    for i in u_labels:
        plt.scatter(df[labels == i, 0], df[labels == i, 1], label=i)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
    plt.legend()
    plt.show()

    #plt.scatter(descriptor[:, 1], descriptor[:, 2])
    # plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s=100, c='red')
    """

    # TODO visualize vocabulary - top 20 patches, 50 random clusters
    # Fjern de clusterene som har få descriptorer?
    centroid_list_ship = getClusterCentres(kmeans_model_ship)
    print("Shape of array with centroids:", np.shape(centroid_list_ship))  # 50xN-dim descriptor
    # print("Example of centroid:", centroid_list[0])

    N = 30
    # computes distances for cluster i
    words_ship = getVocabulary(no_clusters, descriptors_ship, labels_ship, centroid_list_ship, N)

    # Visualize top 30 images from 10 random clusters (from ship class)
    visual_words_ship = []
    for cluster in words_ship:
        visual_words_patch = []
        for index in cluster:
            patch = image_patches_ship[index]
            visual_words_patch.append(patch)
        visual_words_ship.append(visual_words_patch)
    flattened = [val for sublist in visual_words_ship for val in sublist]
    for i in range(10 * N):
        plt.subplot(10, N, i + 1), plt.imshow(flattened[i], cmap='gray')
        plt.xticks([]), plt.yticks([])
    plt.show()

    # Scatterplot

    # Visualize top N image patches from EACH cluster
    # tag = 0
    # for cluster in words:
    #    tag += 1
    #    visual_words_patch = []
    #    for index in cluster:
    #        patch = image_patches[index]
    #        visual_words_patch.append(patch)
    #    fig = plt.figure(num="Figure " + str(tag))
    #    for i in range(N):
    #        plt.subplot(3, 10, i + 1), plt.imshow(visual_words_patch[i], cmap='gray')
    #        plt.xticks([]), plt.yticks([])
    #    plt.show()

    # Show top 1 distances for each cluster
    # visual_words_patch = []
    # for i in range(len(words)):
    #    patch = image_patches[words[i]]
    #    visual_words_patch.append(patch)
    # for i in range(len(visual_words_patch)):
    #    plt.subplot(2, 5, i + 1), plt.imshow(visual_words_patch[i], cmap='gray')
    #    plt.xticks([]), plt.yticks([])
    # plt.show()

    time2 = time.time()
    print('\n', "... Running training time:", (time2 - time1) / 60, "min ...")






















