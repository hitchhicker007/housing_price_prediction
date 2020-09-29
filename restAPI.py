import flask
from flask import request, jsonify
import pickle
import numpy as np

with open('best-model.pkl','rb') as file:
    model = pickle.load(file)


app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/',methods=['GET'])
def home():
    return "<h1>This is API</h1>"

@app.route('/api', methods=['GET'])
def api_size():
    # if 'size' in request.args:
    size = int(request.args['size'])
    typee = int(request.args['type'])
    loc = int(request.args['loc'])
    
    sample = [size,typee,loc]
    sample = np.array(sample).reshape(1,-1)

    result = model.predict(sample)

    return jsonify({'prediction':str(result[0])})

app.run()