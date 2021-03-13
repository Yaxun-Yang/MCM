# Documentation: https://idealo.github.io/imagededup/user_guide/finding_duplicates/
from imagededup.methods import CNN

IMAGE_DIR = "./unlabeled-imgs"

cnn = CNN()
encodings = cnn.encode_images(image_dir=IMAGE_DIR)
duplicates = cnn.find_duplicates(encoding_map=encodings)
# print(duplicates)

duplicate_collection = []
for key, value in duplicates.items():
    if len(value) > 0:
        duplicate_collection.append((key, value))
        print("duplicate pairs:", key, value)
print("number of duplicate pairs:", len(duplicate_collection) / 2)

# Documentation: https://idealo.github.io/imagededup/user_guide/finding_duplicates/#find_duplicates_to_remove
duplicates_to_remove = cnn.find_duplicates_to_remove(
    image_dir=IMAGE_DIR, min_similarity_threshold=0.90
)
print("number of duplicate pairs:", len(duplicates_to_remove))
print(duplicates_to_remove)
