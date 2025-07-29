import cv2
import glob
import joblib
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler

model = joblib.load('detect_cat_dog.z')
scaler = joblib.load("scalar_cat_dog.z")

for i in range(1, 7):
    cat_image = cv2.imread(f'test/Cat/Cat ({i}).jpg', cv2.IMREAD_GRAYSCALE)
    cat_image = cv2.equalizeHist(cat_image)
    cat_image = cv2.resize(cat_image, (64, 64))
    features = hog(cat_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    features_scaled = scaler.transform([features])
    cat_pred = model.predict(features_scaled)
    print(f'Cat{i} should be 0: {cat_pred[0]}')

for i in range(1, 7):
    dog_image = cv2.imread(f'test/Dog/Dog ({i}).jpg', cv2.IMREAD_GRAYSCALE)
    dog_image = cv2.equalizeHist(dog_image)
    dog_image = cv2.resize(dog_image, (64, 64))
    features = hog(dog_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    features_scaled = scaler.transform([features])
    dog_pred = model.predict(features_scaled)
    print(f'Dog{i} should be 1: {dog_pred[0]}')