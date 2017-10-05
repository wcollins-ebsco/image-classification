import os, datetime, hashlib
import urllib, urllib2

import mxnet as mx
import numpy as np
import cv2

from collections import namedtuple
from flask import Flask, request, render_template, redirect, url_for


app = Flask(__name__)
# restrict the size of the file uploaded
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

    
@app.route('/', methods=['POST','GET'])
def index():
    if request.method == 'POST':
        img_url = request.form.get('img_url')
        res = predict(img_url)
        return render_template('index.html', img_src=img_url, prediction_result=res)
    else:
        return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    # check if the post request has the file part
    if request.method != 'POST' or 'file' not in request.files:
        return redirect(url_for('index'))

    f = request.files['file']
    if f.filename == '' or not allowed_file(f.filename):
        return redirect(url_for('index'))

    fn = 'static/img_pool/' \
         + hashlib.sha256(str(datetime.datetime.now())).hexdigest() + f.filename.lower()
    f.save(fn)
    
    res = predict(fn, local=True)
    return render_template('index.html', img_src=fn, prediction_result=res)


Batch = namedtuple('Batch', ['data']) #type=Batch, fieldnames = 'data'

def download(url,prefix=''):
    filename = prefix+url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.urlretrieve(url, filename)

def allowed_file(filename):
    return '.' in filename \
           and filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg', 'bmp']

def get_image(file_location, local=False):
    #local = upload, else url
    if not local:
        file_location = mx.test_utils.download(file_location, dirnm='static/img_pool')

    img = cv2.cvtColor(cv2.imread(file_location), cv2.COLOR_BGR2RGB)
    if img == None:
        return None

    img = cv2.resize(img, (224, 224))
    #convert from (ht x wid x chan) to (batch x chan x wid x ht)
    img = np.swapaxes(img, 0, 2) #hxwxc -> wxcxh
    img = np.swapaxes(img, 1, 2) #wxcxh -> wxhxc
    img = img[np.newaxis, :] #add the batch axis

    return img

def predict(file_location, local=False):#url, mod, synsets):
    img = get_image(file_location, True)
    
    #forward propagate
    mod.forward(Batch([mx.nd.array(img)])) #set image array in field 'data'
    prob = mod.get_outputs()[0].asnumpy() #output is of form [[out1_dev1, out1_dev2], [out2_dev1, out2_dev2]], take first
    prob = np.squeeze(prob) #collapse dimensions of size 1
    
    a = np.argsort(prob)[::-1] #reverse list of indices of sorted values
    result = []
    for i in a[:5]:
        result.append((labels[i].split()[1], prob[i]))
    return result


#download pretrained model params
path='http://data.mxnet.io/models/imagenet-11k/'
download(path+'resnet-152/resnet-152-symbol.json', 'full-')
download(path+'resnet-152/resnet-152-0000.params', 'full-')
download(path+'synset.txt', 'full-')

#read label list, formatted like 'n00005787 benthos'
with open('full-synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]

#load pretrained model
sym, arg_params, aux_params = mx.model.load_checkpoint('full-resnet-152', 0)
#module executes symbol computations
mod = mx.mod.Module(symbol=sym, context=mx.cpu()) #execute predictions on CPU
#init the module, create memory given data_shapes
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))]) #symbols in 'sym' define network size
#init_params() would set random values, instead set pretrained params
mod.set_params(arg_params, aux_params)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
