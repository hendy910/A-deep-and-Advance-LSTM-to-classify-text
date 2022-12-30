#%%
import regex as re
from tensorflow.keras.layers import Dense,LSTM,Dropout,Embedding
from tensorflow.keras import Sequential
from keras.utils.vis_utils import plot_model

def text_cleaning(text):
    """This function removes texts with anomalies such as URLS, @NAME, WASHINGTON (Reuters) 
        and also to convert text into lowercase

    Args:
        text (str): Raw text.

    Returns:
        text (str): Cleaned text
    """

    text = re.sub('@[\S]+'," ",text)
    text = re.sub('bit.ly/\d\w{1,10}',"",text)
    text = re.sub('^.*?\)\s*-', "", text)
    text = re.sub('\[. &?EST\]',"", text)
    text = re.sub('[^a-zA-Z]', ' ', text).lower()

    return text

def lstm_model_creation(num_words,nb_classes, embedded=128,dropout=0.3,num_neurons=64):
    """This function creates LSTM model with embedding layer, 2 LSTM and output

    Args:
        num_words (int): number of vocabulary
        nb_classes (int): number of classes
        embedded (int, optional): number of output of embedding. Defaults to 128.
        dropout (float, optional): The rate dropout. Defaults to 0.3.
        num_neurons (int, optional): Number of Brain Cells. Defaults to 64.

    Returns:
        model: Returns the Model created using sequential 
    """

    model = Sequential()
    model.add(Embedding(num_words,embedded))
    model.add(LSTM(embedded,return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(num_neurons))
    model.add(Dropout(dropout))
    model.add(Dense(nb_classes,activation='softmax'))
    model.summary()

    plot_model(model,show_shapes=True)

    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc'])

    return model

# %%
