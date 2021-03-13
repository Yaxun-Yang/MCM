import os
import cv2
import numpy as np 
import xml.etree.ElementTree as ET

def read_content(xml_file: str):

        tree = ET.parse(xml_file)
        root = tree.getroot()

        list_with_all_boxes = []

        for boxes in root.iter('object'):

            filename = root.find('filename').text

            ymin, xmin, ymax, xmax = None, None, None, None

            ymin = int(boxes.find("bndbox/ymin").text.split(".")[0])
            xmin = int(boxes.find("bndbox/xmin").text.split(".")[0])
            ymax = int(boxes.find("bndbox/ymax").text.split(".")[0])
            xmax = int(boxes.find("bndbox/xmax").text.split(".")[0])

            list_with_single_boxes = (xmin, ymin, xmax, ymax)
            list_with_all_boxes.append(list_with_single_boxes)

        return filename, list_with_all_boxes


cwd = os.getcwd()
print("current working directory:",cwd)

# path_dir_yolo_label = "../annotated_dataset/txt_label/"
path_dir_picture = "../annotated_dataset/JPEGImages/"
path_dir_xml_label = "../annotated_dataset/Annotations/"

image_number = 0

for image_number in range(0, 506):
    if image_number < 10:
        str_img_file_name = f"naver_000{image_number}"
    elif image_number < 100:
        str_img_file_name = f"naver_00{image_number}"
    elif image_number < 1000:
        str_img_file_name = f"naver_0{image_number}"

    # path_file_label_txt = path_dir_yolo_label + str_img_file_name + ".txt"
    path_file_image = path_dir_picture + str_img_file_name + ".jpg"
    path_file_label_xml = path_dir_xml_label + str_img_file_name + ".xml"
    print(path_file_image, path_file_label_xml)

    # read image file
    try: 
        image = cv2.imread(path_file_image)
        # image.show()

        """
        # read text file
        txt_file = open(path_file_label_txt, "r")
        labels = str(txt_file.readlines()[0])
        list_labels = labels.split(" ")

        x_center = list_labels[1]
        y_center = list_labels[2]
        width = list_labels[3]
        height = list_labels[4]
        print("Yolo Format Labels: ", x_center, y_center, width, height)
        """
        
        jpg_name, bounding_boxes = read_content(path_file_label_xml)
        print("JPG File Name:",jpg_name)
        print("XML Format Labels: ", bounding_boxes)

        cropped_path = "../cropped_dataset/"

        
        num = 0
        for box in bounding_boxes:
            x,y,w,h = box
            cropped_img = image[y:h, x:w]
            file_name = f"{str_img_file_name}_{num}.jpg"
            print(file_name, x,y,w,h)
            path_to_save = cropped_path + file_name
            cv2.imwrite(path_to_save, cropped_img)
            num += 1
            # cv2.imshow('cropped_img', cropped_img)
            # cv2.waitKey()
    except:
        pass