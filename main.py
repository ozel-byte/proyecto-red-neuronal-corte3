from datetime import datetime
from matplotlib.cbook import flatten
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pyparsing import java_style_comment
import tensorflow as tf
#import tensorflow_datasets as tfds una musca herramienta misteriosa
import sys
import os
from tensorflow import keras
from keras import layers
from keras.models import Sequential
import seaborn as sns
import serial, time

def main():

    batch_size = 150
    img_height = 100
    img_width = 100
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "dataset/Train",
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        "dataset/validacion",
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names
    print(class_names)
    
    train_ds = train_ds.cache().prefetch(buffer_size=10)
    val_ds = val_ds.cache().prefetch(buffer_size=10)
    num_clases = len(class_names)
    # data_augmentation = tf.keras.Sequential(
    # [
    #     tf.keras.layers.RandomFlip("horizontal",
    #                     input_shape=(img_height,
    #                                 img_width,
    #                                 3)),
    #     tf.keras.layers.RandomRotation(0.1),
    #     tf.keras.layers.RandomZoom(0.1),
    # ]
    # )
    

    # modelo = Sequential([
    # data_augmentation,
    # layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    # layers.Conv2D(32, 3, padding='same', activation='relu'),
    # layers.MaxPooling2D(3,3),
    # layers.Conv2D(64, 3, padding='same', activation='relu'),
    # layers.MaxPooling2D(3,3),
    # layers.Dropout(0.2),
    # layers.Flatten(),
    # layers.Dense(50, activation='relu'),
    # layers.Dense(50, activation='relu'),
    # layers.Dense(num_clases),
    # layers.Activation("softmax")
    # ])
    # modelo.compile(optimizer='adam',
    #           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #           metrics=['accuracy'])
    # modelo.summary()
    # epochs = 200
    # history = modelo.fit(
    # train_ds,
    # validation_data=val_ds,
    # epochs=epochs)
    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']

    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # modelo.save("prueba.h5")

    modelo = tf.keras.models.load_model("new/prueba.h5")
    test_labels = []
    test_images = [] 
    for img, labels in val_ds.take(1):
        test_images.append(img)
        test_labels.append(labels)

    y_pred = np.argmax(modelo.predict(test_images), axis=1).flatten()
    y_true = np.asarray(test_labels).flatten()
    test_acc = sum(y_pred == y_true) / len(y_true)
    print(("Test accuracy: {:.2f}%".format(test_acc * 100)))
    consfusion_matrix = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 10))
    sns.heatmap(consfusion_matrix.numpy(), 
    xticklabels=class_names,
    yticklabels=class_names, 
    annot=True, fmt="d")
    plt.title('Matriz de confusion')
    plt.xlabel('Prediccion')
    plt.ylabel('Real')
    plt.show()
    sensor(modelo,class_names)
    # epochs_range = range(epochs)
    # img             = tf.keras.utils.load_img("foto/img2.jpeg", target_size = (100, 100))
    # img_array       = tf.keras.utils.img_to_array(img)
    # img_array       = tf.expand_dims(img_array, 0)
    # predictions = modelo.predict(img_array)
    # score = tf.nn.softmax(predictions)
    # if 100 * np.max(score) > 30.0 and 100* np.max(score) < 31.0:
    #     print(
    #     "Esta imagen probablemente pertenece a {} con un {:.2f} porcentaje de seguridad."
    #     .format(class_names[np.argmax(score)], 100 * np.max(score)))
    # else:
    #     print("no se reconoce")
    # for i in range(5):
    #     for images, labels in val_ds.take(1):
    #         predictions = modelo.predict(images)
    #         score = tf.nn.softmax(predictions[i])
    #         print(
    #         "Esta imagen probablemente pertenece a {} con un {:.2f} porcentaje de seguridad."
    #         .format(class_names[np.argmax(score)], 100 * np.max(score)))
    #         break

    


    #print(val_ds)
    #y_true = val_ds.map(lambda x: x[1])
    #precision = sum(y_true == y_pred) / len(y_true)
    #print(f'Test set accuracy: {precision:.0%}')
        


def sensor(modelo,class_names):
    arduino = serial.Serial('COM5', 9600)
    while True:
        rawString = arduino.readline()
        if "True" in str(rawString):
            aux = True
            print("entro aqui 1")
            bandera, img_nombre = foto()
            if bandera:
                img             = tf.keras.utils.load_img(img_nombre, target_size = (100, 100))
                img_array       = tf.keras.utils.img_to_array(img)
                img_array       = tf.expand_dims(img_array, 0)
                predictions = modelo.predict(img_array)
                score = tf.nn.softmax(predictions)
                if 100 * np.max(score) > 30.0 and 100* np.max(score) < 31.0:
                    print(
                    "Esta imagen probablemente pertenece a {} con un {:.2f} porcentaje de seguridad."
                    .format(class_names[np.argmax(score)], 100 * np.max(score)))
                else:
                    print("no se reconoce")
            else:
                print("no bandera")
    pass


def foto():
    cap = cv2.VideoCapture(0)
    leido, frame = cap.read()
    nombre = ""
    bandera = False
    if leido == True:
        fecha = str(datetime.now())
        fecha = fecha.replace(".","-")
        fecha = fecha.replace(":","-")
        nombre = f"fotos/{fecha}.png"
        cv2.imwrite(f"fotos/{fecha}.png",frame)
        print("foto tomada correctamente")
        bandera = True
    cap.release()
    return bandera, nombre
    
if __name__ == '__main__':
    main()