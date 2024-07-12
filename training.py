import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Path al tuo dataset
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

from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Caricamento del modello MobileNetV3 Small preaddestrato
base_model = MobileNetV3Small(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Congelare i pesi del modello preaddestrato
base_model.trainable = False

# Aggiungere i propri livelli di classificazione
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(49, activation='softmax')(x)  # num_classes è il numero delle classi nel tuo dataset

# Creazione del modello finale
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Usa sparse_categorical_crossentropy se le etichette sono interi
              metrics=['accuracy'])

history = model.fit(
    train_dataset,
    epochs=10,  # Puoi cambiare il numero di epoche
    validation_data=validation_dataset
)


loss, accuracy = model.evaluate(validation_dataset)
print(f'Validation loss: {loss}')
print(f'Validation accuracy: {accuracy}')
