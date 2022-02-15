import cv2
import os
from xml.etree import ElementTree as et

class ImageProcessing:

    def getFilePath(self, folder):
        """
        Get the path of the images and annotations
        :param folder:
        :return image_path and annotation_path: The paths to the images and annotations
        """
        file_path = []
        directory = os.listdir(folder)
        for filename in directory:
            path = folder + "/" + filename  # "dataset/Optisk/190616_112733.jpg"
            file_path.append(path)

        # array[start_index:end_index:step]
        image_path = file_path[0::2]
        annotation_path = file_path[1::2]

        print("Length image path:", len(image_path))
        print("Length annotation path:", len(annotation_path))

        return image_path, annotation_path


    def readImageInFolder(self, img_path, filter):
        """
        Read the images in folder
        :param img_path: The path to the image
        :return images:
        """
        # img = cv2.imread('dataset/train/ship1.jpg')
        images = []
        for path in img_path:
            img = cv2.imread(path)  # Read image and convert to grayscale
            if filter == True:
                filter_img = cv2.bilateralFilter(img, 20, 75, 75)
                images.append(filter_img)
            else:
                images.append(img)
        return images


    def cropImageFromROI(self, images, label_path):
        """
        Crop the image based on the bounding boxes given in annotation file
        :param images:
        :param label_path: Path to annotation files
        :return cropped_images: The cropped images
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
                    xmin = x.find('bndbox/' + tag_name[0])
                    ymin = x.find('bndbox/' + tag_name[1])
                    xmax = x.find('bndbox/' + tag_name[2])
                    ymax = x.find('bndbox/' + tag_name[3])
                    tag += 1
                    img = images[i]
                    ROI = img[int(float(ymin.text)):int(float(ymax.text)), int(float(xmin.text)):int(float(xmax.text))]
                    # Save image to "result"-folder
                    dir = 'dataset/cropped_images/motorboat_priority'
                    cv2.imwrite(os.path.join(dir, "motorboat_priority" + str(i) + "_" + str(tag) + ".jpg"), ROI)
                    cv2.waitKey(0)
                    cropped_images.append(ROI)

                elif "sailboat with sails down" in name.text:
                    xmin = x.find('bndbox/' + tag_name[0])
                    ymin = x.find('bndbox/' + tag_name[1])
                    xmax = x.find('bndbox/' + tag_name[2])
                    ymax = x.find('bndbox/' + tag_name[3])
                    tag += 1
                    img = images[i]
                    ROI = img[int(float(ymin.text)):int(float(ymax.text)), int(float(xmin.text)):int(float(xmax.text))]
                    # Save image to "result"-folder
                    dir = 'dataset/cropped_images/sailboat_down'
                    cv2.imwrite(os.path.join(dir, "sailboat_down" + str(i) + "_" + str(tag) + ".jpg"), ROI)
                    cv2.waitKey(0)
                    cropped_images.append(ROI)

                elif "sailboat with sails up" in name.text:
                    xmin = x.find('bndbox/' + tag_name[0])
                    ymin = x.find('bndbox/' + tag_name[1])
                    xmax = x.find('bndbox/' + tag_name[2])
                    ymax = x.find('bndbox/' + tag_name[3])
                    tag += 1
                    img = images[i]
                    ROI = img[int(float(ymin.text)):int(float(ymax.text)), int(float(xmin.text)):int(float(xmax.text))]
                    # Save image to "result"-folder
                    dir = 'dataset/cropped_images/sailboat_up'
                    cv2.imwrite(os.path.join(dir, "sailboat_up" + str(i) + "_" + str(tag) + ".jpg"), ROI)
                    cv2.waitKey(0)
                    cropped_images.append(ROI)

                elif "barge" in name.text:
                    xmin = x.find('bndbox/' + tag_name[0])
                    ymin = x.find('bndbox/' + tag_name[1])
                    xmax = x.find('bndbox/' + tag_name[2])
                    ymax = x.find('bndbox/' + tag_name[3])
                    tag += 1
                    img = images[i]
                    ROI = img[int(float(ymin.text)):int(float(ymax.text)), int(float(xmin.text)):int(float(xmax.text))]
                    # Save image to "result"-folder
                    dir = 'dataset/cropped_images/barge'
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
                elif "kayak" in name.text:
                    var = None
                elif "airplane" in name.text:
                    var = None
                elif "helicopter" in name.text:
                    var = None
                else:  # motorboat
                    # print("In motorboat")
                    xmin = x.find('bndbox/' + tag_name[0])
                    ymin = x.find('bndbox/' + tag_name[1])
                    xmax = x.find('bndbox/' + tag_name[2])
                    ymax = x.find('bndbox/' + tag_name[3])
                    tag += 1
                    img = images[i]
                    ROI = img[int(float(ymin.text)):int(float(ymax.text)), int(float(xmin.text)):int(float(xmax.text))]
                    # Save image to "result"-folder
                    dir = 'dataset/cropped_images/motorboat'
                    cv2.imwrite(os.path.join(dir, "motorboat" + str(i) + "_" + str(tag) + ".jpg"), ROI)
                    cv2.waitKey(0)
                    cropped_images.append(ROI)
        return cropped_images