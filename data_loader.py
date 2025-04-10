import os
import cv2

def load_images(path, max_per_class=9999):
    images, labels, class_names = [], [], []
    for idx, class_name in enumerate(sorted(os.listdir(path))):
        folder = os.path.join(path, class_name)
        if not os.path.isdir(folder): continue
        class_names.append(class_name)
        for i, fname in enumerate(os.listdir(folder)):
            if i >= max_per_class: break
            img = cv2.imread(os.path.join(folder, fname))
            if img is not None:
                images.append(img)
                labels.append(idx)
    return images, labels, class_names
