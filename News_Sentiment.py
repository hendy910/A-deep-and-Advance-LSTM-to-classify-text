# %%
import pandas as pd
import regex as re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense,LSTM,Dropout,Embedding
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import json
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import datetime

# %%
df = pd.read_csv('True.csv')
df.head()
print(df['text'][5])
# %%
df.info()
df.head() # nak determine anomalies
df.duplicated().sum()

# %%
df = df.drop_duplicates()

# %%
df.info()

# %%
for index, data in enumerate(df['text']):
    df['text'][index] = re.sub('[^a-zA-Z]',' ',data).lower() # remove punctuations,numbers,lower

# %% Selecting features
text = df['text']
subject = df['subject']

# %% Tokenizer
num_words = 350 # find unique number of words in all the sentences
oov_token = 'Out of Vocab' # out of vocab

tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)

# %% 
tokenizer.fit_on_texts(text)
word_index = tokenizer.word_index
print(dict(list(word_index.items())[0:10]))
# %% to transform the text using tokenizer --> mms.transform
text = tokenizer.texts_to_sequences(text)

# %%
# Padding

padded_text = pad_sequences(text,maxlen=250,padding='post',truncating='pre')

# %%
# One hot encoder
# To instantiate

ohe = OneHotEncoder(sparse=False)

subject = ohe.fit_transform((subject)[::,None])

# %%
# Train Test Split
# expand the dimension before feeding to train_test_split
padded_text = np.expand_dims(padded_text,axis=-1)

X_train,X_test,y_train,y_test = train_test_split(padded_text,subject,test_size=0.2,random_state=123,shuffle=True)

# %% Model Development

embedded = 256

model = Sequential()
model.add(Embedding(num_words,embedded))
model.add(LSTM(embedded,return_sequences=True)) # Input dia kena X_train
model.add(Dropout(0.3))
model.add(LSTM(128))
model.add(Dropout(0.3))
model.add(Dense(2,activation='softmax')) # Output dia kena Y_Train
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['acc'])

# %% TensorBoard Callbacks

LOGS_PATH = os.path.join(os.getcwd(),'logs',datetime.datetime.now().strftime("%Y&m%d-%H%M%S"))
ts_callback = TensorBoard(log_dir=LOGS_PATH)
es_callback = EarlyStopping(monitor='val_loss',patience=5,verbose=0,restore_best_weights=True)

# %% Run
history = model.fit(X_train,y_train,validation_data=(X_test,y_test), batch_size=64,epochs=10,callbacks=[es_callback,ts_callback])

# %% Model Analysis

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training','validation'])
plt.show()

# %%
y_predicted = model.predict(X_test)

# %%
history.history.keys()

# %%

y_predicted = np.argmax(y_predicted,axis=1)
y_test = np.argmax(y_test,axis=1)

print(classification_report (y_test,y_predicted))
print(confusion_matrix(y_test,y_predicted))

# %%
# Model Saving
# to save trained model
model.save('model.h5')

#  to save one hot encoder model

with open('ohe.pkl', 'wb') as f:
    pickle.dump(ohe,f)

# %%
# tokenizer 

token_json = tokenizer.to_json()
with open('tokenizer.json', 'w') as f:
    json.dump(token_json,f)    

# %%
# new_text = input('Insert your review here !!!  ')

new_text = [''' It has been a fascination throughout the first half of the season: how would teams cope with the World Cup break? Manchester City had more players in Qatar than any other Premier League club but there were no signs of weariness in their win at Leeds - and their most important asset looks rested and motivated to fire them to another title defence.

Erling Haaland spoke about his frustration on missing out on the World Cup, with Norway failing to qualify. But the combination of recuperation and determination he's built up over the past six weeks could well pay off for Man City''']

# Need to remove punctuations
for index, data in enumerate(new_text):
    new_text[index] = re.sub('<.*?>','',data) # remove HTML Tags
    new_text[index] = re.sub('[^a-zA-Z]',' ',new_text[index]).lower() # remove punctuations,numbers,lower

new_text = tokenizer.texts_to_sequences(new_text)
padded_new_text = pad_sequences(new_text,maxlen=200,padding='post',truncating='post')

output = model.predict(padded_new_text) # for predict model only recognise number, so use tokenizer

print(ohe.get_feature_names_out())
print(output)
print(ohe.inverse_transform(output))

# %%
