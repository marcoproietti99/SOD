import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Usa una stringa raw per il percorso del dataset
dataset_path = 'C:\\Users\\marco\\OneDrive\\Desktop\\SOD\\images\\images'

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
              loss='sparse_categorical_crossentropy',  # Usa sparse_categorical_crossentropy se le etichette sono interi
              metrics=['accuracy'])

# Addestramento del modello
history = model.fit(
    train_dataset,
    epochs=10,  # Puoi cambiare il numero di epoche
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

# Salva il modello in formato SavedModel
#model.save('C:\\Users\\marco\\OneDrive\\Desktop\\SOD\\my_saved_model.keras')


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

# Percorso del modello salvato
#model_path = 'C:\\Users\\marco\\OneDrive\\Desktop\\SOD\\my_saved_model.keras'

# Carica il modello salvato
#model = tf.keras.models.load_model(model_path)

# Percorso dell'immagine da testare
image_path = 'C:\\Users\\marco\\OneDrive\\Desktop\\SOD\\testprova\\002.jpg'

# Funzione per caricare e preprocessare l'immagine
def load_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0  # Normalizzazione
    image = np.expand_dims(image, axis=0)  # Aggiungi una dimensione per il batch
    return image

# Carica e preprocessa l'immagine
image = load_and_preprocess_image(image_path)

# Fai una predizione
predictions = model.predict(image)
predicted_class_index = np.argmax(predictions, axis=1)[0]

# Carica i nomi delle classi dal file di testo
class_names_path = 'C:\\Users\\marco\\OneDrive\\Desktop\\SOD\\class_names.txt'
with open(class_names_path, 'r') as f:
    class_names = f.read().splitlines()

# Ottieni il nome della classe predetta
predicted_class_name = class_names[predicted_class_index]

# Visualizza l'immagine e le probabilità di predizione per ogni classe
plt.imshow(plt.imread(image_path))
plt.title(f'Predicted class: {predicted_class_name} ({predictions[0][predicted_class_index]*100:.2f}%)')
plt.axis('off')
plt.show()

# Stampa le probabilità per tutte le classi
print("Class probabilities:")
for i, prob in enumerate(predictions[0]):
    print(f"{class_names[i]}: {prob*100:.2f}%")



