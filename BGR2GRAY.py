import cv2
import os

fold_dir = './data/test'
out_dir = './data/test/gray'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
images = [f for f in os.listdir(fold_dir) if f.endswith('.jpg')]
for image in images:
    img = cv2.imread(os.path.join(fold_dir,image))
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(out_dir,image),gray_img)
    print(gray_img.shape)
