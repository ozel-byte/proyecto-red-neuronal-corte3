from copy import copy
from datetime import datetime
from msilib.schema import Dialog
from matplotlib.cbook import flatten
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pyparsing import java_style_comment
import tensorflow as tf
import sys
import os
import cv2 as cv
from tensorflow import keras
from keras import layers
from keras.models import Sequential
import seaborn as sns
import serial, time
import shutil
import threading
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QFileDialog

class Vista(QMainWindow):
    nombre = ""
    def __init__(self):
        super().__init__()
        uic.loadUi ("vista_identificador.ui", self)
        self.btn_imagen.clicked.connect(self.obtener_imagen)
        self.btn_nombre.clicked.connect(self.crear_carpeta)
        self.btn_entrenar.clicked.connect(self.entrenar)
        self.btn_iniciar.clicked.connect(self.iniciar)
        
    def crear_carpeta(self):
        if len(self.nombre_persona.text()) > 0:
            self.nombre = self.nombre_persona.text()
            os.makedirs(f"dataset/test/{self.nombre}", exist_ok=True)
            os.makedirs(f"dataset/Train/{self.nombre}", exist_ok=True)
            os.makedirs(f"dataset/Validacion/{self.nombre}", exist_ok=True)
            self.btn_imagen.setEnabled(True)
            self.btn_nombre.setEnabled(False)
            print("Carpeta creada")
        else:
            print("Campo vacio")
        
    def obtener_imagen(self):
        imagen = QFileDialog.getSaveFileName(None, "Cargar imagen", "", "PNG(*.png);;JPEG(*.jpeg);;JPG(*.jpg);;All Files(*.*)")
        src = cv.imread(imagen[0], cv.COLOR_BAYER_BG2BGR_VNG)
        cv.imshow("Imagen", src)
        shutil.copy(imagen[0], f"dataset/test/{self.nombre}")
        shutil.copy(imagen[0], f"dataset/Train/{self.nombre}")
        shutil.copy(imagen[0], f"dataset/Validacion/{self.nombre}")
        print(imagen)
        print("\n", imagen[0])
        
    def entrenar(self):
        self.btn_imagen.setEnabled(False)
        self.btn_nombre.setEnabled(True)
        main(2)
    
    def iniciar(args):
        main(1)

def main(bandera):
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

    if bandera == 1:
        print("Bandera: ", bandera)
        if os.path.exists("new/prueba.h5") == False:
            pass
            modelo = crear_red(img_height, img_width, num_clases, train_ds, val_ds)
        else:
            modelo = tf.keras.models.load_model("new/prueba.h5")
    else:
        modelo = crear_red(img_height, img_width, num_clases, train_ds, val_ds)
    
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
    fig2 = plt.figure(figsize=(12, 7))
    sns.heatmap(consfusion_matrix.numpy(), 
    xticklabels=class_names,
    yticklabels=class_names, 
    annot=True, fmt="d")
    plt.title('Matriz de confusion')
    plt.xlabel('Prediccion')
    plt.ylabel('Real')
    plt.show()
    t_start = threading.Thread(target=sensor, args=(modelo, class_names, ))
    t_start.start()
        

def crear_red(img_height, img_width, num_clases, train_ds, val_ds):
    data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal",
                        input_shape=(img_height,
                                    img_width,
                                    3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ]
    )
    
    modelo = Sequential([
        data_augmentation,
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(3,3),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(3,3),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(50, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(num_clases),
        layers.Activation("softmax")
    ])
    modelo.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    modelo.summary()
    epochs = 200
    history = modelo.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    fig = plt.figure(figsize=(12, 7))
    plt.plot(history.history['loss'], label='Evolucion')
    plt.plot(history.history['val_loss'], label = 'val_Evolucion')
    plt.xlabel('Epoca')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    modelo.save("new/prueba.h5")
    return modelo

def sensor(modelo,class_names):
    arduino = serial.Serial('COM3', 9600)
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
                print(
                    "Esta imagen probablemente pertenece a {} con un {:.2f} porcentaje de seguridad."
                    .format(class_names[np.argmax(score)], 100 * np.max(score)))
                if 100 * np.max(score) > 20:
                    print(
                    "Esta imagen probablemente pertenece a {} con un {:.2f} porcentaje de seguridad."
                    .format(class_names[np.argmax(score)], 100 * np.max(score)))
                else:
                    print("no se reconoce")
            else:
                print("no bandera")


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
    app = QApplication(sys.argv)
    ventana = Vista()
    ventana.show()
    sys.exit(app.exec_())