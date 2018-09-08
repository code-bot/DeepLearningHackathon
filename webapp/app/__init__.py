import os
from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash
from datetime import datetime
import json

app = Flask(__name__) # create the application instance
emojis = ["smiley","sad"]
score = 0
counter = 1
currentImage = None

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/game')
def game():
	return render_template('game.html', firstEmoji=emojis[0])

@app.route('/snapshot', methods=['POST'])
def snapshot():
	print(request.files)
	global currentImage
	if request.files.get('webcam'):
		currentImage = request.files.get('webcam')
		return 'Received file'
	else:
		return 'No file'

@app.route('/nextEmoji', methods=['GET'])
def nextEmoji():
	global counter
	global emojis
	if counter >= len(emojis):
		return 'No Emoji'
	emoji = emojis[counter]
	counter += 1
	return emoji

@app.route('/getScore', methods=['GET'])
def getScore():
	global currentImage
	global score
	if currentImage != None:
		score += calcScore(currentImage)
		currentImage = None
		return str(score)
	else:
		return 'No Score'

def calcScore(image):
	return 1