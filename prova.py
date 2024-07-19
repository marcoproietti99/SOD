import tensorflow as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report

# Funzione per estrarre l'etichetta dal nome del file
def extract_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    filename = parts[-1]
    
    # Suddivide il filename usando '_'
    split_filename = tf.strings.split(filename, '_')
    
    # Prende tutti i pezzi fino al primo numero
    class_name_parts = tf.strings.regex_replace(split_filename, r'\d+', '')  # Rimuove i numeri
    class_name_parts = tf.strings.regex_replace(class_name_parts, r'\.jpeg|\.jpg|\.png', '')  # Rimuove le estensioni
    
    # class_name = tf.strings.reduce_join(class_name_parts, separator=' ')
    class_name = tf.strings.regex_replace(class_name_parts, r'_', '')  # Rimuove spazi duplicati
    class_name = tf.strings.strip(class_name)  # Rimuove eventuali spazi iniziali o finali
    
    return class_name

# Percorso del dataset
dataset_path = 'C:\\Users\\marco\\OneDrive\\Desktop\\SOD\\images\\images'

# Lista dei file nel dataset
file_list = tf.data.Dataset.list_files(os.path.join(dataset_path, '*/*'))

# Funzione per caricare e preprocessare un'immagine
def load_and_preprocess_image(file_path):
    # Leggi il file immagine
    image = tf.io.read_file(file_path)
    # Decodifica dell'immagine JPEG
    image = tf.image.decode_jpeg(image, channels=3)
    # Ridimensionamento dell'immagine
    image = tf.image.resize(image, [224, 224])
    # Normalizzazione dell'immagine
    image = image / 255.0
    return image

# Funzione per creare un dataset di coppie (immagine, etichetta)
def process_file(file_path):
    # Estrai l'etichetta dal nome del file
    label = extract_label(file_path)
    # Carica e preprocessa l'immagine
    image = load_and_preprocess_image(file_path)
    return image, label

# Mappa la funzione di elaborazione su ogni file nel dataset
dataset = file_list.map(process_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Estrazione e ordinamento dei nomi delle classi
def get_class_names(dataset):
    # Estrai solo le etichette
    labels = dataset.map(lambda image, label: label)
    # Usa un dataset di etichette uniche
    unique_labels = labels.batch(1).map(lambda x: tf.unique(tf.reshape(x, [-1]))[0])
    unique_labels = unique_labels.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    unique_labels = unique_labels.batch(1).map(lambda x: x[0])
    unique_labels = list(unique_labels.as_numpy_iterator())
    unique_labels = sorted(set([l.decode('utf-8') for l in unique_labels]))
    return unique_labels

class_names = get_class_names(dataset)
class_name_to_index = {name: index for index, name in enumerate(class_names)}

# Funzione per convertire le etichette in indice numerico
def convert_label_to_index(image, label):
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(class_names),
            values=tf.range(len(class_names), dtype=tf.int64)
        ),
        default_value=-1
    )
    label_index = table.lookup(label)
    return image, label_index

# Converti le etichette in indici numerici
dataset = dataset.map(lambda x, y: tf.py_function(convert_label_to_index, [x, y], [tf.float32, tf.int64]), num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Dividi il dataset in training e validation
dataset_size = sum(1 for _ in file_list)
train_size = int(0.8 * dataset_size)
train_dataset = dataset.take(train_size)
validation_dataset = dataset.skip(train_size)

# Batch e mescola il dataset di training
batch_size = 32
train_dataset = train_dataset.shuffle(buffer_size=train_size)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Batch il dataset di validation
validation_dataset = validation_dataset.batch(batch_size)

# Numero di classi
num_classes = len(class_names)

# Caricamento del modello MobileNetV3 Small preaddestrato
base_model = tf.keras.applications.MobileNetV3Small(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Congelare i pesi del modello preaddestrato
base_model.trainable = False

# Aggiungere i propri livelli di classificazione
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

# Creazione del modello finale
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

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

# Generazione della matrice di confusione
val_labels = []
val_predictions = []
for images, labels in validation_dataset:
    val_labels.extend(labels)
    predictions = model.predict(images)
    val_predictions.extend(tf.argmax(predictions, axis=1))

val_labels = np.array(val_labels)
val_predictions = np.array(val_predictions)

# Stampa della matrice di confusione
cm = tf.math.confusion_matrix(val_labels, val_predictions)
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






