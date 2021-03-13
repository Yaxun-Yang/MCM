from PIL import Image

im1 = Image.open("/Users/noopy/murder_hornet_dataset/pn-ng/test2.png")
rgb_im = im1.convert("RGB")
rgb_im.save("/Users/noopy/murder_hornet_dataset/pn-ng/test2.jpg")
