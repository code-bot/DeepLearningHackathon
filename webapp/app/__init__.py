import os
from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash
from datetime import datetime
import json
from PIL import Image
import binascii
import io
import numpy as np
import cv2
import randomcode as rc

app = Flask(__name__) # create the application instance
emojis = ["happy","sad"]
score = 0
counter = 1
currentImage = []

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
		filestr = request.files.get('webcam').read()
		npimg = np.fromstring(filestr, np.uint8)
		currentImage = cv2.imdecode(npimg, 1)
		currentImage = cv2.resize(currentImage, (96,96))
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
	global counter
	if len(currentImage) > 0:
		score += calcScore(currentImage, counter)
		currentImage = []
		return str(score)
	else:
		return 'No Score'

def calcScore(image, stype):
	return rc.get_score(image.mean(axis=2), stype)