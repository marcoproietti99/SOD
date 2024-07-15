import tensorflow as tf
import os

# Funzione per estrarre l'etichetta dal nome del file
def extract_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    filename = parts[-1]
    
    class_name = ''
    for part in tf.strings.split(filename, '_'):
        if tf.strings.regex_full_match(part, '[a-zA-Z]+'):
            class_name = tf.strings.join([class_name, part], separator='_')
        else:
            break
    
    return tf.strings.strip(class_name, '_')

# Percorso del dataset
dataset_path = 'C:\\Users\\marco\\OneDrive\\Desktop\\SOD\\resized'

# Lista dei file nel dataset
file_list = tf.data.Dataset.list_files(dataset_path + '/*/*')

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
dataset = file_list.map(process_file)

# Dividi il dataset in training e validation
train_size = int(0.8 * len(file_list))
train_dataset = dataset.take(train_size)
validation_dataset = dataset.skip(train_size)

# Batch e mescola il dataset di training
batch_size = 32
train_dataset = train_dataset.shuffle(buffer_size=train_size)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Batch il dataset di validation
validation_dataset = validation_dataset.batch(batch_size)

# Stampa dei nomi delle classi
class_names = sorted(set(label.numpy().decode() for _, label in dataset))
print("Class names: ", class_names)

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
print(tf.keras.metrics.classification_report(val_labels, val_predictions, target_names=class_names))

