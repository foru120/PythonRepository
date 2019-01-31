from PIL import Image
import os

MAX_SIZE = (0, 0)  # width, height
MAX_CAP = 0

for root, dirs, files in os.walk("G:/04_dataset/eye_verification/eye_dataset_v2/train"):
    for file in files:
        img = Image.open(os.path.join(root, file))
        cap = os.path.getsize(os.path.join(root, file))

        if MAX_SIZE[0] < img.size[0]:  # width 만 비교
            MAX_SIZE = img.size

        if MAX_CAP < cap:
            MAX_CAP = cap

print('File Max Size :', MAX_SIZE)
print('File Max Cap  :', round(MAX_CAP / 1024.), '(KB)')