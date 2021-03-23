from flask import request
from main import app
import pandas as pd
from flask_cors import cross_origin


@app.route('/')
def index():
    return {"response": True}
