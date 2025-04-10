import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_images
from sift_extractor import extract_sift
from vocab_builder import build_vocab
from bow_builder import create_bow
from evaluation import plot_confusion_matrix, plot_class_metrics

# ==== 1. 加载数据 ====
print("📥 Loading images...")
data_path = "./data"  # 修改为你的数据集路径
images, labels, class_names = load_images(data_path)

# ==== 2. 提取 SIFT 特征 ====
print("🔍 Extracting SIFT descriptors...")
descriptors_list = extract_sift(images)

# ==== 3. 构建视觉词袋 ====
print("📦 Building vocabulary (Bag of Words)...")
kmeans = build_vocab(descriptors_list, clusters=200)

# ==== 4. 构建 BoW 特征向量 ====
print("🧱 Creating BoW vectors...")
X = create_bow(descriptors_list, kmeans)
y = np.array(labels)[:len(X)]

# ==== 5. 特征标准化 ====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==== 6. 训练 + 测试模型 ====
print("🎯 Training + Evaluating model...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

# ==== 7. 打印报告 & 可视化 ====
print(classification_report(y_test, y_pred, target_names=class_names))

# 混淆矩阵
plot_confusion_matrix(y_test, y_pred, class_names)

# 每类 precision/recall/F1 柱状图
plot_class_metrics(y_test, y_pred, class_names)
