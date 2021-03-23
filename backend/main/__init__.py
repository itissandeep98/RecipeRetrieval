from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.secret_key = '5ZN5zi!45QUsGG'
app.config.from_pyfile('config.cfg')

from main import routes