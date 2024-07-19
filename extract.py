import tensorflow as tf
import os

def extract_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    filename = parts[-1]
    
    # Suddivide il filename usando '_'
    split_filename = tf.strings.split(filename, '_')
    
    # Prende tutti i pezzi fino al primo numero
    class_name_parts = tf.strings.regex_replace(split_filename, r'\d+', '')  # Rimuove i numeri
    class_name_parts = tf.strings.regex_replace(class_name_parts, r'\.jpeg|\.jpg|\.png', '')  # Rimuove le estensioni
    
    class_name = tf.strings.reduce_join(class_name_parts, separator='_')
    class_name = tf.strings.regex_replace(class_name, r'_+', '_')  # Rimuove duplicati '_'
    class_name = tf.strings.strip(class_name, '_')  # Rimuove eventuali '_' iniziali o finali
    
    return class_name

# Funzione per ottenere tutti i file immagine in una directory e sottodirectory
def get_all_image_files(directory):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpeg', '.jpg', '.png')):
                image_files.append(os.path.join(root, file))
    return image_files

# Percorso alla directory delle immagini
image_directory = 'C:\\Users\\marco\\OneDrive\\Desktop\\SOD\\images\\images'

# Ottiene tutti i file immagine
image_files = get_all_image_files(image_directory)

# Estrae le etichette per ciascun file immagine
labels = [extract_label(image_file).numpy().decode('utf-8').rstrip('_') for image_file in image_files]

# Stampa le etichette
for image_file, label in zip(image_files, labels):
    print(f"File: {image_file}, Label: {label}")

