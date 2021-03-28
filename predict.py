import tensorflow as tf
import tensorflow_hub as hub
import argparse
import numpy as np
import json
from PIL import Image

def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)
    image /= 255
    image = tf.image.resize(image ,(224, 224))
    return image.numpy()


def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)[0]
    sorted_predictions = sorted(predictions, reverse=True)
    probs = sorted_predictions[0:top_k]
    classes = np.argsort(predictions)[-top_k:][::-1] + 1
    classes = np.char.mod('%d', classes)
    return probs, classes;


parser = argparse.ArgumentParser(
    description='Prediction program',
)

parser.add_argument('image_path', action="store")
parser.add_argument('model_path', action="store")
parser.add_argument('--top_k', action="store", type=int, default=5)
parser.add_argument('--category_names', action="store", default='label_map.json')

args = parser.parse_args()
# TODO: Load the Keras model
saved_keras_model_filepath = './saved_model.h5'

reloaded_keras_model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer':hub.KerasLayer});

predictions = predict(args.image_path, reloaded_keras_model, args.top_k)

with open(args.category_names, 'r') as f:
    class_names = json.load(f)

for prob, classKey in zip(predictions[0], predictions[1]):
    class_name = class_names[classKey]
    print(class_name + ': ' + str(prob * 100) + '%')