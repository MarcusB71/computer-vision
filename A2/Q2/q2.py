import pandas
import numpy as np
import cv2
import glob
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from skimage.feature import hog

# DATA
data = []
labels = []

for i, address in enumerate(glob.glob('train/*/*')):
    img = cv2.imread(address, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    img = cv2.equalizeHist(img)
    features = hog(img, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    data.append(features)
    
    # Label: 0 for cat, 1 for dog
    if "cat" in address.lower():
        labels.append(0)
    elif "dog" in address.lower():
        labels.append(1)

    if i%200==0:
        print(f'[INFO] {i} images processed')

data = np.array(data)
labels = np.array(labels)

print(data.shape)
print(labels)

X = data
y = labels

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)
joblib.dump(scaler, "scalar_cat_dog.z")

# pca = PCA(n_components=100)
# X_reduced = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# MODEL
# KNN
# for k in [1, 3, 5, 7, 9]:
#     model = KNeighborsClassifier(n_neighbors=k)
#     model.fit(X_train, y_train)
#     # EVALUATE
#     preds = model.predict(X_test)
#     print(f"k={k}, accuracy={accuracy_score(y_test, preds):.4f}")
#     joblib.dump(model, "detect_cat_dog.z")

# Logistic Regression 65-70% accuracy
for c in [0.01, 0.1, 1, 10, 100]:
    model = LogisticRegression(C=c, max_iter=1000)
    model.fit(X_train, y_train)
    # EVALUATE
    preds = model.predict(X_test)
    print(f"C={c}, accuracy={accuracy_score(y_test, preds):.4f}")
    joblib.dump(model, "detect_cat_dog.z")

# SGD
# model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
# model.fit(X_train, y_train)
# preds = model.predict(X_test)
# print("SGDClassifier Accuracy:", accuracy_score(y_test, preds))

for i in range(1, 6):
    cat_image = cv2.imread(f'test/Cat/Cat ({i}).jpg', cv2.IMREAD_GRAYSCALE)
    cat_image = cv2.equalizeHist(cat_image)
    cat_image = cv2.resize(cat_image, (64, 64))
    features = hog(cat_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    features_scaled = scaler.transform([features])
    cat_pred = model.predict(features_scaled)
    print(f'Cat{i} should be 0: {cat_pred[0]}')
