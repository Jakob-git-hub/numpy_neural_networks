# 3. Modell kompilieren
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Daten laden und vorverarbeiten
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalisierung der Pixelwerte auf den Bereich [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Modellarchitektur (Nur Dense-Layers)
model = models.Sequential([
    # Flacht das 28x28 Bild in einen 784-dimensionalen Vektor ab
    layers.Flatten(input_shape=(28, 28)), 
    
    # Erste versteckte Schicht mit ReLU und Dropout gegen Overfitting
    layers.Dense(256, activation='relu'),

    
    # Zweite versteckte Schicht für tiefere Merkmalskombinationen
    layers.Dense(128, activation='relu'),

    
    # Ausgangsschicht für die 10 Bekleidungsklassen
    layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Modell trainieren
# 20 Epochs reichen in der Regel völlig aus, um >86% zu erreichen
history = model.fit(
    x_train, y_train, 
    epochs=20, 
    batch_size=8, 
    validation_split=0.1
)

# 5. Evaluierung auf den Testdaten
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nErziehlte Test-Genauigkeit: {test_acc:.4f}')