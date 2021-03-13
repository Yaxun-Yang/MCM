import os
import glob


def delete_given_file(image_name):

    file_name = image_name.split(".")[0]
    IMG_PATH = "./annotated_dataset/img"
    TXT_PATH = "./annotated_dataset/txt_label"
    XML_PATH = "./annotated_dataset/xml_label"

    img_file = f"{IMG_PATH}/{file_name}.jpg"
    txt_file = f"{TXT_PATH}/{file_name}.txt"
    xml_file = f"{XML_PATH}/{file_name}.xml"

    files_list = [img_file, txt_file, xml_file]
    int_files = len(files_list)
    try:
        for item in files_list:
            os.remove(item)
        if int_files == 3:
            print(f"removed {int_files} file of {file_name}")
    except:
        print(f"FAIL: removing {file_name} failed")


def remove_img_file(image_name):
    file_name = image_name.split(".")[0]
    IMG_PATH = "./unlabeled-imgs"
    img_file = f"{IMG_PATH}/{file_name}.jpg"
    try:
        os.remove(img_file)
        print(f"removed {img_file} file")
    except:
        print(f"FAIL: removing {img_file} failed")


# delete_given_file("naver_0074.jpg")
removable_file_list = [
    "naver_0615.jpg",
    "naver_0508.jpg",
    "naver_0353.jpg",
    "naver_0677.jpg",
    "naver_0592.jpg",
    "naver_0443.jpg",
    "naver_0904.jpg",
    "naver_0460.jpg",
    "naver_0832.jpg",
    "naver_0388.jpg",
    "naver_0408.jpg",
    "naver_0513.jpg",
    "naver_0429.jpg",
]


for file_item in removable_file_list:
    # delete_given_file(file_item)
    remove_img_file(file_item)
