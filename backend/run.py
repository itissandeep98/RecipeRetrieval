from main import app
from main.models import InvertedIndex
from main.models import tf_idfmatrices

if __name__ == '__main__':
	app.run(debug = True, port = '5000')
