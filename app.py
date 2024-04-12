import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#import the required libraries and Load the Universal Sentence Encoder's TF Hub module

import pandas as pd       # To work with tables
import tensorflow as tf   # To work with USE4
import tensorflow_hub as hub # contains USE4

from numpy import dot     # to calculate the dot product of two vectors
from numpy.linalg import norm  #for finding the norm of a vector

from flask import Flask,render_template,request

module_url = "https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2"
model = hub.load(module_url)
def embed(input):
  return model(input)

Data = pd.read_csv(r"C:\Users\dshem\OneDrive\Desktop\STS\STS-1\sts_data.csv")

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
        #Finding Cosine similarity
        ans = []     # This list will contain the cosin similarity value for each vector pair present.                                                     
        for i in range(len(ip)):
            messages = [ip['text1'][i], ip['text2'][i]]  #storing each sentence pair in messages            
            message_embeddings = embed(messages)         #converting the sentence pair to vector pair using the embed() function                    
            a = tf.make_ndarray(tf.make_tensor_proto(message_embeddings))   #storing the vector in the form of numpy array
            cos_sin = dot(a[0], a[1])/(norm(a[0])*norm(a[1]))   #Finding the cosine between the two vectors          
            ans.append(cos_sin)            #Appending the values into the ans list
        Ans = pd.DataFrame(ans, columns = ['similarity score'])
        ip = ip.join(Ans)
        ip['similarity score'] = ip['similarity score'].apply(lambda x: round(x, 1))
        result=ip[['similarity score']].to_dict("records")[0]
    return render_template('result.html',result=result)

if __name__ == '__main__':
	app.run(debug=True)