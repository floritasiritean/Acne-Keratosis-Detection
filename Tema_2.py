import os
import numpy as np
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Definiți funcția pentru a încărca imaginile dintr-un folder
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = io.imread(img_path)
        img_resized = resize(img, (50, 50))  # Redimensionați imaginile la o dimensiune fixă
        images.append(img_resized.flatten())
        labels.append(folder.split('-')[-1])  # Adăugați eticheta în funcție de numele folderului
    return images, labels

# Încărcați imaginile de antrenare
folder1_path = 'dataset/acnee'
folder2_path = 'dataset/cheratoza'

images1, labels1 = load_images_from_folder(folder1_path)
images2, labels2 = load_images_from_folder(folder2_path)

# Combinați imaginile și etichetele din ambele foldere
all_images = np.concatenate([images1, images2], axis=0)
all_labels = np.concatenate([labels1, labels2])

# Împărțirea inițială a datelor în antrenare și restul
X_train, X_temp, y_train, y_temp = train_test_split(all_images, all_labels, test_size=0.3, random_state=42)

# Împărțirea restului în validare și testare
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

# Standardizarea datelor pentru KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Evaluarea modelului KNN pe setul de validare
knn_valid_predictions = knn_model.predict(X_valid_scaled)
knn_valid_accuracy = metrics.accuracy_score(y_valid, knn_valid_predictions)
print(f'Accuracy of KNN on validation set: {knn_valid_accuracy}')

# Evaluarea modelului KNN pe setul de testare
knn_test_predictions = knn_model.predict(X_test_scaled)
knn_test_accuracy = metrics.accuracy_score(y_test, knn_test_predictions)
print(f'Accuracy of KNN on test set: {knn_test_accuracy}')

# Naive Bayes (NB)
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Evaluarea modelului NB pe setul de validare
nb_valid_predictions = nb_model.predict(X_valid)
nb_valid_accuracy = metrics.accuracy_score(y_valid, nb_valid_predictions)
print(f'Accuracy of Naive Bayes on validation set: {nb_valid_accuracy}')

# Evaluarea modelului NB pe setul de testare
nb_test_predictions = nb_model.predict(X_test)
nb_test_accuracy = metrics.accuracy_score(y_test, nb_test_predictions)
print(f'Accuracy of Naive Bayes on test set: {nb_test_accuracy}')

# Definiți funcția pentru afișarea imaginilor din setul de testare
def display_images_with_labels(images, labels, original_size=(50, 50, 3)):
    for i in range(len(images)):
        image = images[i].reshape(*original_size)  # Redimensionați imaginea la dimensiunea originală
        label = labels[i]

        plt.imshow(image)
        plt.title(f'Eticheta: {label}')
        plt.show()
        time.sleep(2)  # Adăugați o pauză de 2 secunde între afișarea imaginilor

# Afișați toate imaginile din setul de testare cu etichetele corespunzătoare
display_images_with_labels(X_test, y_test)



# KNN
knn_predictions = knn_model.predict(X_test)
knn_conf_matrix = confusion_matrix(y_test, knn_predictions)

nb_predictions = nb_model.predict(X_test)
nb_conf_matrix = confusion_matrix(y_test, nb_predictions)

# Funcție pentru afișarea matricei de confuzie
def plot_confusion_matrix(conf_matrix, class_names):
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Calculați matricea de confuzie pentru KNN
knn_conf_matrix = confusion_matrix(y_test, knn_predictions)
# Afișați matricea de confuzie pentru KNN
plot_confusion_matrix(knn_conf_matrix, class_names=["Acnee-KNN", "Cheratoză-KNN"])

# Calculați matricea de confuzie pentru Naive Bayes
nb_conf_matrix = confusion_matrix(y_test, nb_predictions)
# Afișați matricea de confuzie pentru Naive Bayes
plot_confusion_matrix(nb_conf_matrix, class_names=["Acnee-NB", "Cheratoză-NB"])

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Definirea setului de parametri pentru căutarea grid
knn_param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}

# Inițializarea clasificatorului KNN
knn_classifier = KNeighborsClassifier()

# Inițializarea căutării grid pentru KNN
knn_grid_search = GridSearchCV(knn_classifier, knn_param_grid, cv=5)

# Antrenarea modelului pe setul de antrenare
knn_grid_search.fit(X_train, y_train)

# Afișarea celui mai bun parametru găsit
print("Cel mai bun număr de vecini pentru KNN:", knn_grid_search.best_params_['n_neighbors'])

# Antrenarea KNN cu cel mai bun parametru
best_knn = KNeighborsClassifier(n_neighbors=knn_grid_search.best_params_['n_neighbors'])
best_knn.fit(X_train, y_train)

# Evaluarea performanței pe setul de validare și setul de testare
knn_val_accuracy = best_knn.score(X_valid, y_valid)
knn_test_accuracy = best_knn.score(X_test, y_test)

# Afișarea rezultatelor
print(f"Precizie pe setul de validare pentru KNN: {knn_val_accuracy}")
print(f"Precizie pe setul de testare pentru KNN: {knn_test_accuracy}")


# Optimizare pentru Naive Bayes
param_grid_nb = {'var_smoothing': np.logspace(0,-9, num=100)}
nb_classifier = GaussianNB()
nb_grid_search = GridSearchCV(nb_classifier, param_grid_nb, cv=5)
nb_grid_search.fit(X_train, y_train)

# Afișează cel mai bun parametru pentru Naive Bayes
print("Cel mai bun parametru pentru Naive Bayes:", nb_grid_search.best_params_['var_smoothing'])

# Antrenează Naive Bayes cu cel mai bun parametru
best_nb = GaussianNB(var_smoothing=nb_grid_search.best_params_['var_smoothing'])
best_nb.fit(X_train, y_train)

# Evaluate the model on validation set and test set
nb_val_accuracy_optimized = best_nb.score(X_valid, y_valid)
nb_test_accuracy_optimized = best_nb.score(X_test, y_test)

# Afișează rezultatele pentru Naive Bayes optimizat
print(f"Optimizare Naive Bayes - Precizie pe setul de validare: {nb_val_accuracy_optimized}")
print(f"Optimizare Naive Bayes - Precizie pe setul de testare: {nb_test_accuracy_optimized}")


# Combină etichetele din seturile de antrenare, validare și testare
all_labels_combined = np.concatenate([y_train, y_valid, y_test])

# Creează histograma
plt.figure(figsize=(8, 6))
sns.countplot(x=all_labels_combined)
plt.title('Histograma Distribuției Etichetelor')
plt.xlabel('Etichete')
plt.ylabel('Număr de Exemple')
plt.show()