import random

train_and_val_txt = "./resized_dataset/ImageSets/Main/trainval.txt"

imgs = open(train_and_val_txt).read().splitlines()
print(len(imgs))
test_dataset_proportion = 0.4
test_imgs_number = int(len(imgs) * test_dataset_proportion)
val_imgs_data = random.sample(imgs, test_imgs_number)
print(val_imgs_data)

train_imgs_data = [item for item in imgs if item not in val_imgs_data]


# write train.txt
with open("resized_dataset/ImageSets/Main/train.txt", "w") as f:
    for item in train_imgs_data:
        f.write("%s\n" % item)

# write val.txt
with open("resized_dataset/ImageSets/Main/val.txt", "w") as f:
    for item in val_imgs_data:
        f.write("%s\n" % item)
