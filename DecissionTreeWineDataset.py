# %%
# Import library
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree

# %%
# Load wine dataset
wine = load_wine()
df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
df_wine['target'] = wine.target

# Tampilkan deskripsi statistik
print(df_wine.describe().T)

# %%
# Tampilkan beberapa data awal
print(df_wine.head())

# %%
# Visualisasi data dengan pairplot
sns.pairplot(df_wine, hue='target', palette='Set2')  # optional
plt.show()

# %%
# Split data training dan testing
x = df_wine.drop('target', axis=1)
y = df_wine['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# %%
# Training model
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Prediksi
y_pred = model.predict(x_test)

# %%
# Evaluasi model
print("Classification Report:\n", classification_report(y_test, y_pred))

# %%
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Wine Dataset")
plt.show()

# %%
# Visualisasi decision tree
fig, ax = plt.subplots(figsize=(25, 20))
tree.plot_tree(model, feature_names=wine.feature_names, class_names=wine.target_names, filled=True)
plt.show()

# %%
# Uji coba prediksi data baru
wine_test_data = {
    'alcohol': 13.0,
    'malic_acid': 2.0,
    'ash': 2.5,
    'alcalinity_of_ash': 15.0,
    'magnesium': 100.0,
    'total_phenols': 2.0,
    'flavanoids': 2.5,
    'nonflavanoid_phenols': 0.3,
    'proanthocyanins': 1.5,
    'color_intensity': 5.0,
    'hue': 1.0,
    'od280/od315_of_diluted_wines': 3.0,
    'proline': 1000.0
}
input_df = pd.DataFrame([wine_test_data])
prediction = model.predict(input_df[wine.feature_names])
print("Prediksi kelas wine:", prediction)
