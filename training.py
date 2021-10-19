import random
import json
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np

lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intents.json").read())

words = []
classes = []
documents = []
igonre_letters = ['?', '!', '.', ',']

for intent in intents["intents"]:
	for pattern in intent["patterns"]:
		word_list = nltk.word_tokenize(pattern)
		words.extend(word_list)
		documents.append((word_list, intent["tag"]))
		if intent["tag"] not in classes:
			classes.append(intent["tag"])

words = [lemmatizer.lemmatize(word) for word in words if word not in igonre_letters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

training = []
output_empty = [0] * len(classes)

train_x = []
train_y = []

for document in documents:
	bag = []
	word_pattern = document[0]
	word_pattern = [lemmatizer.lemmatize(word.lower()) for word in word_pattern]
	for word in words:
		bag.append(1) if word in word_pattern else bag.append(0)	
	output_row = list(output_empty)
	output_row[classes.index(document[1])] = 1
	
	train_x.append(bag)
	train_y.append(output_row)


train_x = np.array(train_x)
train_y = np.array(train_y)

model = Sequential()
model.add(Dense(128,
				input_shape=(train_x.shape[1],),
				activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(64,
				activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(len(train_y[0]),
				activation="softmax"))

model.compile(loss="categorical_crossentropy",
			optimizer="adam",
			metrics=["accuracy"])

hist = model.fit(train_x,
				train_y,
				epochs=200,
				batch_size=5)

model.save("scylla_chatbot.h5", hist)
print("Done!")

