from flask import Flask, make_response, render_template, request, session, send_file
from flask_cors import CORS, cross_origin
import sys, io, random
import numpy as np

from utils import im, binary
from api import encode, decode

from PIL import Image
import matplotlib.pyplot as plt

import IPython

app = Flask(__name__)
CORS(app)#, resources={r"/static/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type';

@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Cache-Control, Authorization, X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, DELETE')
    return response


@app.route('/encode', methods = ['POST'])
def upload_request():
   if request.method == 'POST':
      content = request.files['file']
      encoded = encode_file(content, request.form.get("data"))
      response = make_response(encoded)
      response.headers["Content-Disposition"] = "attachment; filename=protected_{}".format(content.filename)
      response.headers["Content-type"] = content.mimetype
      response.headers["Cache-Control"] = "must-revalidate"
      response.headers["Pragma"] = "must-revalidate"
      return response

@app.route('/decode', methods = ['POST'])
def decode_request():
   if request.method == 'POST':
      content = request.files['file']
      return decode_file(content)

def encode_file(file, data):

    bitcode = [int(digit) for digit in bin(int(data))[2:].zfill(64)]
    ids = set([line[:-1] for line in open("ids.txt")])
    print (ids, file=sys.stderr)
    print (binary.str(bitcode) not in ids, file=sys.stderr)
    if binary.str(bitcode) not in ids:
      with open("ids.txt", "a+") as f:
        f.write(binary.str(bitcode)+"\n")

    print (bitcode, file=sys.stderr)
    bitcode = bitcode[0:32]

    image = plt.imread(file)/255.0
    print (image.min())
    encoded = encode(image, bitcode)

    im.save(encoded, "static/final_encoded.jpg")
    return open("static/final_encoded.jpg", 'rb').read()

def decode_file(file):
    image = plt.imread(file)/255.0
    code = decode(image)

    ids = [binary.parse(line[:-1]) for line in open("ids.txt")]
    print (binary.str(code))
    def match(x):
      return binary.distance(x[0:32], code)
    best_id = min(ids, key=match)
    print (binary.str(best_id), match(best_id))
    code2 = code + best_id[32:64]
    print (len(code2))
    fix = 0
    for i in random.sample(list(range(0, 32)), 32):
      if code2[i] != best_id[i]:
        code2[i] = best_id[i]
        fix += 1
        if fix == 8: break

    print (binary.str(code2), file=sys.stderr)
    data = str(int(binary.str(code2), 2))
    return data

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)

