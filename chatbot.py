import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intents.json").read())

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

model = load_model("scylla_chatbot.h5")

def bag_of_words(sentence):
	bag = [0] * len(words)

	s_words = nltk.word_tokenize(sentence)
	s_words = [lemmatizer.lemmatize(word.lower()) for word in s_words]

	for s in s_words:
		for i, w in enumerate(words):
			if w == s:
				bag[i] = 1
	return np.array(bag)

def chat():
	print("Start talking(type quit to stop)!")
	run = True
	while run:
		sentence = input("You: ")
		if sentence.lower() == "quit":
			print("Scylla: Bye")
			break
		bow = bag_of_words(sentence)
		results = model.predict(np.array([bow]))[0]
		result_index = np.argmax(results)
		tag = classes[result_index]

		if results[result_index] > 0.7:
			for tg in intents["intents"]:
				if tg["tag"] == tag:
					if tg["tag"] == "goodbye":
						responses = tg["responses"]
						run = False
					else:
						responses = tg["responses"]

			print("Scylla: ", random.choice(responses))
		else:
			tag = "noanswer"
			for tg in intents["intents"]:
				if tg["tag"] == tag:
					responses = tg["responses"]
			print("Scylla:", random.choice(responses))


chat()