import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding,RNN,LSTM,Dense
from keras import Sequential
import numpy as np
import math
import pandas as pd

df=pd.read_csv(r"Fake.csv",names=["title","text","sub","date"])

x=[]
y=[]


for i in range(1,501):
    x.append(df.text[i])
    y.append([0,1])
    
x_train=np.array(x[0:400])
x_test=np.array(x[400:500])
y_train=np.array(y[0:400])
y_test=np.array(y[400:500])


df=pd.read_csv(r"\True.csv",names=["title","text","sub","date"])


x=[]
y=[]
for i in range(1,501):
    x.append(df.text[i])
    y.append([1,0])

x_train=np.append(x_train,x[0:400],axis=0)
x_test=np.append(x_test,x[400:500],axis=0)

print(x_train.shape)

y_train=np.append(y_train,y[0:400],axis=0)
y_test=np.append(y_test,y[400:500],axis=0)


vocab_size=50

encoded_docs=[one_hot(d,vocab_size) for d in np.array(x_train)]
padded_docs=pad_sequences(encoded_docs,100,padding='post')

print(padded_docs)
embed_dim = 128
lstm_out = 200
#batch_size = 32

model = Sequential()
model.add(Embedding(embed_dim, embed_dim,input_length = padded_docs.shape[1], dropout = 0.2))
model.add(LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2))

model.add(Dense(2,activation='softmax')) 
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics = ['accuracy'])
print(model.summary())

encoded_test=[one_hot(d,vocab_size) for d in x_test]
padded_test=pad_sequences(encoded_test,100,padding='post')

# fit model
model.fit(padded_docs, y_train, epochs=2, verbose=0)

#test model
y=model.predict(padded_test)
err=0
#print(y_test)
l=y_test.shape[0]

for i in range(l):
    if y[i][0]>y[i][1]:
        y[i][0]=1
        y[i][1]=0
    else:
        y[i][0]=0
        y[i][1]=1
    err+=(y_test[i][0]-y[i][0]+y_test[i][0]-y[i][0])/2
err=err/l
acc=1-err        
print("Accuracy: ",acc*100,"%")
input=None
print(input)
inp=input("Enter A News To Test: ")
inp=np.array([inp])
encoded_test=[one_hot(d,vocab_size) for d in inp]
padded_test=pad_sequences(encoded_test,100,padding='post')
output=model.predict(padded_test)
if(output[0][0]>output[0][1]):
    print("True News!!! Wow")
else:
    print("Oops!!! Fake News")
