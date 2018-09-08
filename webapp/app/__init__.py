import os
from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash
from datetime import datetime
import json

app = Flask(__name__) # create the application instance

@app.route('/')
def home():

    
	return render_template('home.html')

@app.route('/sahaj')
def sahaj():

	return render_template('sahaj.html')