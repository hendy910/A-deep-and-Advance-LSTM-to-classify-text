# %%
import pandas as pd
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
import pickle
import json
from modules import text_cleaning, lstm_model_creation

# %% Data loading
CSV_PATH = os.path.join(os.getcwd(),'True.csv')
df = pd.read_csv(CSV_PATH)

# %% Data inspection
df.info()
df.duplicated().sum()

# %%
print(df['text'][85]) 
# have URL, have @Donald Trump, $ Sign, filter WASHINGTON (Reuters) : New Header, [1901 EST]

# %%
for index,temp in enumerate(df['text']):
    df['text'][index] = text_cleaning(temp)

# %%
print(df['text'][85]) 
# %% features selection

X = df['text']
y = df['subject']

# %% Data Preprocessing
# Tokenizer

num_words = 5000
tokenizer = Tokenizer(num_words=num_words,oov_token='<OOV>')
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)

# %%
# Padding
X = pad_sequences(X,maxlen=300,padding='post',truncating='post')

# %%
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y[::,None])

# %%
# Train test Split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123,shuffle=True)


# %%
model = lstm_model_creation(num_words,y.shape[1])

# %%
history = model.fit(X_train,y_train,epochs=5,batch_size=64,validation_data=(X_test,y_test))
# %%


y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(y_test,axis=1)

print(classification_report(y_test,y_pred))
cm = confusion_matrix(y_test,y_pred)

# %%
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot()
# %%
model.save('model2.h5')

# %%
#  to save one hot encoder model


with open('ohe2.pkl', 'wb') as f:
    pickle.dump(ohe,f)

# %%
# tokenizer 

token_json = tokenizer.to_json()
with open('tokenizer2.json', 'w') as f:
    json.dump(token_json,f)    
