import tensorflow as tf
import numpy as np

count = 0
outputMapping = {}

with open("labels/labels.txt") as file:
    for line in file:
        outputMapping[count] = line.strip('\n')
        count += 1

model = tf.keras.models.load_model('models/network.model')

print("Please provide a board to classifice moves for")
boardString = input()

singluarInputs = boardString.split(",")
input = np.zeros(31)

for i in range(0, 31):
    input[i] = float(singluarInputs[i])

input = tf.keras.utils.normalize(input)
predictions = model.predict([input])

topTenVals = np.sort(predictions[0])[-10:]
topTen = np.argsort(predictions[0])[-10:]
print("------------------------")

for i in range(9, -1, -1):
    print(str(abs(i-10)) + ": " +
          str(outputMapping[topTen[i]]) + " : " + str(topTenVals[i]))
    print("------------------------")
