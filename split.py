import os
import shutil
from sklearn.model_selection import train_test_split

# Definisci il percorso del dataset originale
original_dataset_dir = 'C:\\Users\\marco\\OneDrive\\Desktop\\SOD\\images\\images'

# Definisci il percorso per il set di costruzione (training e validation)
construction_dataset_dir = 'C:\\Users\\marco\\OneDrive\\Desktop\\SOD\\train'
train_dir = os.path.join(construction_dataset_dir, 'train')
val_dir = os.path.join(construction_dataset_dir, 'val')

# Definisci il percorso per il set di test
test_dataset_dir = 'C:\\Users\\marco\\OneDrive\\Desktop\\SOD\\test'

# Creare le directory di destinazione
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dataset_dir, exist_ok=True)

# Dividere i dati
for class_name in os.listdir(original_dataset_dir):
    class_dir = os.path.join(original_dataset_dir, class_name)
    if os.path.isdir(class_dir):
        images = os.listdir(class_dir)
        train_val_images, test_images = train_test_split(images, test_size=0.2, random_state=42)
        train_images, val_images = train_test_split(train_val_images, test_size=0.2, random_state=42)

        # Copiare i file nel set di training
        train_class_dir = os.path.join(train_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        for image in train_images:
            shutil.copy(os.path.join(class_dir, image), os.path.join(train_class_dir, image))

        # Copiare i file nel set di validation
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(val_class_dir, exist_ok=True)
        for image in val_images:
            shutil.copy(os.path.join(class_dir, image), os.path.join(val_class_dir, image))

        # Copiare i file nel set di test
        test_class_dir = os.path.join(test_dataset_dir, class_name)
        os.makedirs(test_class_dir, exist_ok=True)
        for image in test_images:
            shutil.copy(os.path.join(class_dir, image), os.path.join(test_class_dir, image))
