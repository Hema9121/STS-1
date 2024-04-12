import pandas as pd 
import pickle
import tensorflow as tf
import tensorflow_hub as hub

from numpy import dot                                  
from numpy.linalg import norm

Data = pd.read_csv(r"C:\Users\dshem\OneDrive\Desktop\STS\STS-1\sts_data.csv")

module_url = "https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2" 
model = hub.load(module_url)
"""def embed(input):
  return model(input)
"""
message = [Data['text1'][0], Data['text2'][0]]
model(message)
# message_embeddings = embed(message)

filename = 'sts_model.pkl'
pickle.dump(model, open(filename, 'wb'))

"""ans = []                                                       
for i in range(len(Data)):
  messages = [Data['text1'][i], Data['text2'][i]]               
  message_embeddings = model(messages)                          
  a = tf.make_ndarray(tf.make_tensor_proto(message_embeddings)) 
  cos_sin = dot(a[0], a[1])/(norm(a[0])*norm(a[1]))             
  ans.append(cos_sin)
"""