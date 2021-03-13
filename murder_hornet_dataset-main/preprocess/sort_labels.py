import os
import glob
import shutil

unlabeled_path = "./unlabeled-imgs"


def sort_xml_txt_img_and_move(ROOT_PATH):
    IMG_PATH = "./annotated_dataset/img"
    TXT_PATH = "./annotated_dataset/txt_label"
    XML_PATH = "./annotated_dataset/xml_label"

    jpg_file_list = glob.glob(f"{ROOT_PATH}/*.jpg")
    txt_file_list = glob.glob(f"{ROOT_PATH}/*.txt")
    xml_file_list = glob.glob(f"{ROOT_PATH}/*.xml")

    for image_name in jpg_file_list:
        # print(image_name)
        file_path = image_name.split(".")[1]
        # print(file_path)
        file_name = file_path.split("/")[-1]
        img_file = f"{ROOT_PATH}/{file_name}.jpg"
        txt_file = f"{ROOT_PATH}/{file_name}.txt"
        xml_file = f"{ROOT_PATH}/{file_name}.xml"
        if os.path.isfile(txt_file) & os.path.isfile(xml_file):
            print(file_name, "label file exists")
            shutil.move(img_file, IMG_PATH)
            shutil.move(txt_file, TXT_PATH)
            shutil.move(xml_file, XML_PATH)
            print(f"moved items for {file_name} to 3 corresponding paths")

        else:
            print(file_name, "is missing some label")
            if not os.path.isfile(txt_file):
                print(file_name, "missing txt label")
            if not os.path.isfile(xml_file):
                print(file_name, "missing xml label")


sort_xml_txt_img_and_move(unlabeled_path)
