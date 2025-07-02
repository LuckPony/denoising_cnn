import cv2
import os

fold_dir = 'F:/Data/archive/Ground_truth'
out_dir = './data/train/groundTruth'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
images = [f for f in os.listdir(fold_dir) if f.endswith('.jpg')]
i = 1
for image in images:
    img = cv2.imread(os.path.join(fold_dir,image))
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(out_dir,image),gray_img)
    print(f'当前进度：{i}/{len(images)}')
    i += 1
