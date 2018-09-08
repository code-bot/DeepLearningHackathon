import os
from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash
from datetime import datetime
import json

app = Flask(__name__) # create the application instance

@app.route('/')
def home():

    
	return render_template('home.html')

@app.route('/game')
def game():

	return render_template('game.html')

@app.route('/webcam')
def webcam():
	return render_template('webcam.html')

@app.route('/snapshot', methods=['POST'])
def snapshot():
	if request.method == 'POST':
		print(request.files)
		return 'Received file'