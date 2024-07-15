import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Percorso del modello salvato
model_path = 'C:\\Users\\marco\\OneDrive\\Desktop\\SOD\\my_saved_model.keras'

# Carica il modello salvato
model = tf.keras.models.load_model(model_path)

# Percorso dell'immagine da testare
image_path = 'C:\\Users\\marco\\OneDrive\\Desktop\\SOD\\testprova\\004.jpg'




# Funzione per caricare e preprocessare l'immagine
def load_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0  # Normalizzazione
    image = np.expand_dims(image, axis=0)  # Aggiungi una dimensione per il batch
    return image

loss, accuracy = model.evaluate(image)
print(f'Validation loss: {loss}')
print(f'Validation accuracy: {accuracy}')
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


