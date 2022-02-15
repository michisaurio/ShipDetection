import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.image import extract_patches_2d
import os


def getSIFTfeatures(image, extractor):
    """
    Given an image, compute SIFT feature descriptors for the image and return them.
    :param image: Input image
    :param extractor:
    :return: keypoints and descriptors
    """
    kp, des = extractor.detectAndCompute(image, None)
    return kp, des

def showKeypointsOnImg(img, kp):
    img_kp = cv2.drawKeypoints(img, kp, img, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #img_kp = cv2.drawKeypoints(img, kp, img, color=(0,255,0), flags=None)
    plt.imshow(img_kp)
    plt.show()



def explainKeypoint(kp):
    print('this is an example of a single SIFT keypoint:\n* * *')
    print('angle\n', kp.angle) # computed orientation of the keypoint (-1 if not applicable); it's in [0,360) degrees
    # and measured relative to image coordinate system (y-axis is directed downward), i.e in clockwise.

    print('\nclass_id\n', kp.class_id) # object class (if the keypoints need to be clustered by an object they belong to)

    print('\noctave (image scale where feature is strongest)\n', kp.octave) # octave (pyramid layer) from which the keypoint has been extracted.

    print('\npt (x,y)\n', kp.pt) # coordinates of the keypoint [x,y]

    print('\nresponse\n', kp.response) #  the response by which the most strong keypoints have been selected. Can be used for further sorting or subsampling

    print('\nsize (diameter)\n', kp.size) # diameter of the meaningful keypoint neighborhood
    print('* * *')


def getImagePatches(img, random_state, patch_size, n_patches):
    """
    Extracts subimages
    :param img_file: path for an image
    :param random_state:
    :param patch_size: size of each patch
    :param n_patches: number of patches to be extracted
    :return:
    """
    # Extract sub-images
    patch = extract_patches_2d(img,
                               patch_size=patch_size,
                               max_patches=n_patches,
                               random_state=random_state)

    return patch.reshape((n_patches, np.prod(patch_size) * len(img.shape)))


def printDescriptor(des):
    print("Descriptor list for whole image has shape:", des.shape)
    print('SIFT descriptors are vectors of shape', des[0].shape)
    print('They look like this:', des[0], '\n')


def visualizeDescriptor(des):
    # visualized another way:
    plt.imshow(des.reshape(16, 8))
    # plt.imshow(descriptors[0].reshape(128, 1))
    plt.show()



def generateClusters(descriptors, n_clusters):
    """
    given vocabulary, and number of clusters in which vocabulary is to be divided, applies kmeans algorithm and returns

    :param vocab: vocabulary containing feature descriptors from all the images available for training.
    :param n_clusters: Number of clusters for dividing vocab.
    :return: kmeans - which is further used to predict.
    """
    kmeans = KMeans(n_clusters).fit(descriptors)
    return kmeans


def getClusterCentres(kmeans):
    centroid = kmeans.cluster_centers_
    return centroid


def plotCodewords(bool, titles, nr_clusters, rows, columns):
    # Visualizing codewords/visual words
    if bool is True:
        for i in range(nr_clusters):
            plt.subplot(rows, columns, i + 1), plt.imshow(codewords[i].reshape(16, 8))
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()
    else:
        for i in range(nr_clusters):
            plt.subplot(rows, columns, i + 1), plt.imshow(codewords[i].reshape(16, 8))
            plt.xticks([]), plt.yticks([])
        plt.show()


def saveImagePatches(kp, directory):
    # image patch
    x = 0
    y = 1
    for ind in range(len(kp)):
        ymin = kp[ind].pt[y] - (kp[ind].size / 2)
        ymax = kp[ind].pt[y] + (kp[ind].size / 2)
        xmin = kp[ind].pt[x] - (kp[ind].size / 2)
        xmax = kp[ind].pt[x] + (kp[ind].size / 2)
        image_patch = img[int(float(ymin)):int(float(ymax)), int(float(xmin)):int(float(xmax))]
        dir = directory
        cv2.imwrite(os.path.join(dir, "image_patch" + str(ind) + ".jpg"), cv2.cvtColor(image_patch, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)



if __name__ == '__main__':
    img = cv2.imread('Dataset/ship4.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # defining feature extractor that we want to use
    extractor = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)

    # Detecting features and extracting descriptor
    keypoints, descriptors = getSIFTfeatures(img, extractor)


    print('Number of keypoints on image: ', len(keypoints))
    print('Number of descriptors:', len(descriptors), '\n')

    plt.scatter(descriptors[10, :], descriptors[7000, :])
    plt.title('2 SIFT descriptors')
    plt.show()

    #explainKeypoint(keypoints[100])

    #showKeypointsOnImg(img, keypoints)

    #saveImagePatches(keypoints, "imagePatch")

    #visualizeDescriptor(descriptors[0])

    nr_clusters = 50
    kmeans = generateClusters(descriptors, nr_clusters)
    codewords = getClusterCentres(kmeans)

    #plt.scatter(descriptors[10,:], descriptors[7000,:], c=kmeans.labels_)
    #plt.title('2 SIFT components and its labels')
    #plt.axis('off')
    #plt.show()



    titles = ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5", "Cluster 6", "Cluster 7", "Cluster 8",
              "Cluster 9", "Cluster 10"]
    plotCodewords(False, titles, nr_clusters, rows=5, columns=10)







