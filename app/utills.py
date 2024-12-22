import os
import numpy as np
import cv2

def load_data(data_dir):
    images, labels = [], []
    for label, label_idx in zip(['glioma', 'meningioma', 'notumor', 'pituitary'], range(4)):
        label_dir = os.path.join(data_dir, label)
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(label_idx)

    X = np.array(images).reshape(-1, 128, 128, 1) / 255.0
    y = np.array(labels)
    return X, y
