import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Usa una stringa raw per il percorso del dataset
dataset_path = 'C:\\Users\\marco\\OneDrive\\Desktop\\SOD\\images\\images'
dataset_path_test= 'C:\\Users\\marco\\OneDrive\\Desktop\\SOD\\test2'

# Caricamento del dataset
train_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),  # MobileNetV3 Small usa immagini 224x224
    batch_size=32
)

validation_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)


# Stampa dei nomi delle classi
class_names = train_dataset.class_names
print("Class names: ", class_names)

output_path = 'C:\\Users\\marco\\OneDrive\\Desktop\\SOD\\class_names.txt'

# Salva i nomi delle classi in un file di testo
with open(output_path, 'w') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")

# Numero di classi
num_classes = len(class_names)

# Caricamento del modello MobileNetV3 Small preaddestrato
base_model = MobileNetV3Small(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Congelare i pesi del modello preaddestrato
base_model.trainable = False

# Aggiungere i propri livelli di classificazione
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Creazione del modello finale
model = Model(inputs=base_model.input, outputs=predictions)

# Compilazione del modello
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Addestramento del modello
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=validation_dataset
)

# Valutazione del modello
loss, accuracy = model.evaluate(validation_dataset)
print(f'Validation loss: {loss}')
print(f'Validation accuracy: {accuracy}')

# Prevedi le etichette del set di validazione
val_labels = []
val_predictions = []
for images, labels in validation_dataset:
    val_labels.extend(labels)
    predictions = model.predict(images)
    val_predictions.extend(np.argmax(predictions, axis=1))

val_labels = np.array(val_labels)
val_predictions = np.array(val_predictions)

# Generazione della matrice di confusione
cm = confusion_matrix(val_labels, val_predictions)
print('Confusion Matrix')
print(cm)

# Visualizzazione della matrice di confusione
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Report di classificazione
print('Classification Report')
print(classification_report(val_labels, val_predictions, target_names=class_names))


loss, accuracy = model.evaluate(test_dataset)


# Prevedi le etichette del set di validazione
test_labels = []
test_predictions = []
for images, labels in test_dataset:
    test_labels.extend(labels)
    predictions = model.predict(images)
    test_predictions.extend(np.argmax(predictions, axis=1))

test_labels = np.array(test_labels)
test_predictions = np.array(test_predictions)

