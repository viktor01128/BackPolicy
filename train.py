import tensorflow as tf
from numpy import genfromtxt

trainingX = genfromtxt('datasets/trainingX.csv', delimiter=',')
trainingY = genfromtxt('datasets/trainingY.csv')

trainingX = tf.keras.utils.normalize(trainingX, axis=1)

testX = genfromtxt('datasets/testX.csv', delimiter=',')
testY = genfromtxt('datasets/testY.csv')

testX = tf.keras.utils.normalize(testX, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, input_dim=31, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(601, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[
              'accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10)])

model.fit(trainingX, trainingY, epochs=10,
          batch_size=32, validation_split=0.15)

model.evaluate(testX, testY)

model.save('models/network.model')
