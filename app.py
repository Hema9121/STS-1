import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

from numpy import dot
from numpy.linalg import norm

from flask import Flask,render_template,url_for,request

module_url = "https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2"
model = hub.load(module_url)
def embed(input):
  return model(input)

Data = pd.read_csv(r"C:\Users\dshem\OneDrive\Desktop\STS\STS-1\sts_data.csv")
#message = [Data['text1'][0], Data['text2'][0]]
#message_embeddings = embed(message)
#message_embeddings

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        text1,text2=message[0],message[1]
        dict1={}
        for i in ["text1","text2"]:
            dict1[i]=eval(i)
        ip = pd.DataFrame(dict1, index=[0])
        ans = []                                                        
        for i in range(len(ip)):
            messages = [ip['text1'][i], ip['text2'][i]]              
            message_embeddings = embed(messages)                          
            a = tf.make_ndarray(tf.make_tensor_proto(message_embeddings)) 
            cos_sin = dot(a[0], a[1])/(norm(a[0])*norm(a[1]))             
            ans.append(cos_sin)
        Ans = pd.DataFrame(ans, columns = ['similarity score'])
        ip = ip.join(Ans)
        #ip['similarity score'] = ip['similarity score'] + 1
        #ip['similarity score'] = ip['similarity score']/ip['similarity score'].abs().max()
        ip['similarity score'] = ip['similarity score'].apply(lambda x: round(x, 1))
        result=ip[['similarity score']].to_dict("list")
    return render_template('result.html',result=result)

if __name__ == '__main__':
	app.run(debug=True)