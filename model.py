import tensorflow as tf
import numpy as np

# Asegúrate de cargar correctamente tu modelo preentrenado
model = tf.keras.models.load_model('models/animal_classifier_transfer_learning.keras')

class_labels = [
    "dog", "horse", "elephant", "butterfly", "chicken",
    "cat", "cow", "sheep", "squirrel", "spider"
]
def preprocess_image(uploads):
    """
    Preprocesa la imagen para que sea compatible con el modelo.
    """
    img = tf.keras.preprocessing.image.load_img(uploads, target_size=(64, 64))  # Ajusta según el tamaño de entrada de tu modelo
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)  # Añade una dimensión adicional para el batch
    img_array = img_array / 255.0  # Normaliza entre 0 y 1 si tu modelo lo requiere
    return img_array

def predicti(uploads):
    """
    Realiza predicciones utilizando el modelo cargado.
    """
    img_array = preprocess_image(uploads)  # Preprocesa la imagen
    predictions = model.predict(img_array)  # Obtiene las predicciones (probabilidades)

    # Decodifica las predicciones utilizando las etiquetas de clase
    decoded_predictions = [
        {"label": class_labels[i], "probability": float(predictions[0][i])}
        for i in range(len(class_labels))
    ]

    # Ordena las predicciones por probabilidad en orden descendente
    decoded_predictions = sorted(decoded_predictions, key=lambda x: x['probability'], reverse=True)

    return decoded_predictions


