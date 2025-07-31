from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPool2D, Dense,MaxPooling2D
from keras.layers import AveragePooling2D, Flatten, Activation, Bidirectional
from keras.layers import BatchNormalization, Dropout
from keras.layers import Concatenate, Add, Multiply, Lambda
from keras.layers import UpSampling2D, Reshape
from keras.layers import Add
from keras.layers import Concatenate
from keras.layers import Reshape
from keras.models import Model
from keras.layers import LSTM,GRU
import tensorflow as tf
from keras import backend as K
import keras
import random
from keras import backend as K
from tensorflow.keras.backend import ctc_batch_cost
from inference import process_uploaded_image

import warnings
import numpy as np
import itertools

from tensorflow.keras.utils import plot_model


warnings.filterwarnings("ignore")

letters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZÖÜ'

#image height
img_h=90
#image width
img_w=160
#image Channels
# img_c=1
# classes for softmax with number of letters +1 for blank space in ctc
num_classes=len(letters)+1
batch_size=64
max_length=15 # considering max length of ground truths labels to be 15



def ctc_loss_function(args):
    """
    CTC loss function takes the values passed from the model returns the CTC loss using Keras Backend ctc_batch_cost function
    """
    y_pred, y_true, input_length, label_length = args 
    # since the first couple outputs of the RNN tend to be garbage we need to discard them, found this from other CRNN approaches
    # I Tried by including these outputs but the results turned out to be very bad and got very low accuracies on prediction 
    y_pred = y_pred[:, 2:, :]
    return ctc_batch_cost(y_true, y_pred, input_length, label_length)  

def encode_words_labels(word):
    """
    Encodes the Ground Truth Labels to a list of Values like eg.HAT returns [17,10,29]
    """
    label_lst=[]
    for char in word:
        label_lst.append(letters.find(char)) # keeping 0 for blank and for padding labels
    return label_lst


def words_from_labels(labels):
    """
    converts the list of encoded integer labels to word strings like eg. [12,10,29] returns CAT 
    """
    txt=[]
    for ele in labels:
        if ele == len(letters): # CTC blank space
            txt.append("")
        else:
            #print(letters[ele])
            txt.append(letters[ele])
    return "".join(txt)


def model_definition(stage, drop_out_rate=0.35):
    
    model_input=Input(shape=(90, 160, 1),name='img_input',dtype='float32')

    # Convolution layer 
    model = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(model_input) 
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = MaxPooling2D(pool_size=(2, 2), name='max1')(model) 

    model = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(model) 
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = MaxPooling2D(pool_size=(2, 2), name='max2')(model) 

    model = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(model) 
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(model)
    model=Dropout(drop_out_rate)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = MaxPooling2D(pool_size=(1, 2), name='max3')(model)  

    model = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(model) 
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(512, (3, 3), padding='same', name='conv6')(model)
    model=Dropout(drop_out_rate)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = MaxPooling2D(pool_size=(1, 2), name='max4')(model) 

    model = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(model)
    model=Dropout(0.25)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)    

    # CNN to RNN
    model = Reshape(target_shape=((88, 1280)), name='reshape')(model)  
    # model = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(model)  
    model = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(model)  

    # RNN layer
    model=Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='he_normal'), merge_mode='sum')(model)
    model=Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='he_normal'), merge_mode='concat')(model)

    # transforms RNN output to character activations:
    model = Dense(num_classes, kernel_initializer='he_normal',name='dense2')(model) 
    y_pred = Activation('softmax', name='softmax')(model)

    
    labels = Input(name='ground_truth_labels', shape=[max_length], dtype='float32') 
    input_length = Input(name='input_length', shape=[1], dtype='int64') 
    label_length = Input(name='label_length', shape=[1], dtype='int64') 

    #CTC loss function
    loss_out = Lambda(ctc_loss_function, output_shape=(1,),name='ctc')([y_pred, labels, input_length, label_length]) #(None, 1)

    if stage=='train':
        return model_input,y_pred,Model(inputs=[model_input, labels, input_length, label_length], outputs=loss_out)
    else:
        return Model(inputs=[model_input], outputs=y_pred)        


def accuracies(actual_labels,predicted_labels,is_train):
    """
    Takes a List of Actual Outputs, predicted Outputs and returns their accuracy and letter accuracy across
    all the labels in the list
    """
    accuracy=0
    letter_acc=0
    letter_cnt=0
    count=0
    for i in range(len(actual_labels)):
        predicted_output=predicted_labels[i]
        actual_output=actual_labels[i]
        count+=1
        for j in range(min(len(predicted_output),len(actual_output))):
            if predicted_output[j]==actual_output[j]:
                letter_acc+=1
        letter_cnt+=max(len(predicted_output),len(actual_output))
        if actual_output==predicted_output:
            accuracy+=1
    final_accuracy=np.round((accuracy/len(actual_labels))*100,2)
    final_letter_acc=np.round((letter_acc/letter_cnt)*100,2)
    return final_accuracy,final_letter_acc

def decode_batch(test_func, word_batch):
    """
    Takes the Batch of Predictions and decodes the Predictions by Best Path Decoding and Returns the Output
    """
    out = test_func([word_batch])[0] #returns the predicted output matrix of the model
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = words_from_labels(out_best)
        ret.append(outstr)
    return ret

class VizCallback(keras.callbacks.Callback):
    """
    The Custom Callback created for printing the Accuracy and Letter Accuracy Metrics at the End of Each Epoch
    """

    def __init__(self, test_func, text_img_gen,is_train,acc_compute_batches):
        self.test_func = test_func
        self.text_img_gen = text_img_gen
        self.is_train=is_train                #used to indicate whether the callback is called to for Train or Validation Data
        self.acc_batches=acc_compute_batches  # Number of Batches for which the metrics are computed typically equal to steps/epoch

    def show_accuracy_metrics(self,num_batches):
        """
        Calculates the accuracy and letter accuracy for each batch of inputs, 
        and prints the avarage accuracy and letter accuracy across all the batches
        """
        accuracy=0
        letter_accuracy=0
        batches_cnt=num_batches
        while batches_cnt>0:
            word_batch = next(self.text_img_gen)[0]   #Gets the next batch from the Data generator
            decoded_res = decode_batch(self.test_func,word_batch['img_input'])
            actual_res=word_batch['source_str']
            acc,let_acc=accuracies(actual_res,decoded_res,self.is_train)
            accuracy+=acc
            letter_accuracy+=let_acc
            batches_cnt-=1
        accuracy=accuracy/num_batches
        letter_accuracy=letter_accuracy/num_batches
        if self.is_train:
            print("Train Average Accuracy of "+str(num_batches)+" Batches: ",np.round(accuracy,2)," %")
            print("Train Average Letter Accuracy of "+str(num_batches)+" Batches: ",np.round(letter_accuracy,2)," %")
        else:
            print("Validation Average Accuracy of "+str(num_batches)+" Batches: ",np.round(accuracy,2)," %")
            print("Validation Average Letter Accuracy of "+str(num_batches)+" Batches: ",np.round(letter_accuracy,2)," %")
            
        
    def on_epoch_end(self, epoch, logs={}):
        self.show_accuracy_metrics(self.acc_batches)


model_input,y_pred,img_text_recog=model_definition('train')
# img_text_recog.summary()
plot_model(img_text_recog, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

from keras.callbacks import EarlyStopping,ModelCheckpoint
early_stop=EarlyStopping(monitor='val_loss',patience=2,restore_best_weights=True)
model_chk_pt=ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.weights.h5', monitor = "val_loss",
  save_weights_only = True
  )
import os
import datetime

from keras import optimizers
adam=optimizers.Adam()
logdir = os.path.join("logs_127", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
#Creating a Dummy Loss function as in Keras there is no CTC loss implementation which actually takes 4 inputs 
#The loss function in keras accepts only 2 inputs, so create a dummy loss which is a work around for implementing CTC in Keras
#The Actual loss computation happens in ctc_loss_function defined above
img_text_recog.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)

# %%
import pickle

def pad_array(arr, target_length):

    num_samples = len(arr)
    padded_arr = np.zeros((num_samples, target_length))
    for i in range(num_samples):
        padded_arr[i] = np.pad(arr[i], (0, target_length - len(arr[i])), mode='constant')
    return padded_arr
def flatten_concatenation(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list
# %%
with open('training_data.pkl', "rb") as fp:
    X_train_mat = pickle.load(fp)
    # df_t = pd.DataFrame(X_train_mat)
X_train_ls = [tup[0]  for tup in X_train_mat]
y_train_ls = [tup[1]  for tup in X_train_mat]
X_train = np.array(flatten_concatenation(X_train_ls))
X_train = X_train[..., np.newaxis]
 
y_train = flatten_concatenation(y_train_ls)
y_train_enc = [np.array(encode_words_labels(label)) for label in y_train]
y_train_enc = pad_array(y_train_enc, 15)


with open('testing_data.pkl', "rb") as fp:
    X_test_mat = pickle.load(fp)
X_test_ls = [tup[0]  for tup in X_test_mat]
y_test_ls = [tup[1]  for tup in X_test_mat]
X_test = np.stack(flatten_concatenation(X_test_ls), axis=0)
X_test = X_test[..., np.newaxis]
y_test = flatten_concatenation(y_test_ls)
y_test_enc = [np.array(encode_words_labels(label)) for label in y_test]
y_test_enc = pad_array(y_test_enc, 15)

# %%
import logging
logging.basicConfig(filename='/tmp/myapp.log', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger=logging.getLogger(__name__)



# model_input, labels, input_length, label_length


# test_func = K.function([model_input], [y_pred])
# viz_cb_train = VizCallback( test_func, train_gene.next_batch(),True,train_num_batches)
# input_length = np.ones((batch_size, 1)) * 15
# label_length = np.zeros((batch_size, 1))
# input_lengths_train = np.ones((X_train.shape[0], 1)) * 44  # based on your reshape layer
# label_lengths_train = np.array([[len(lbl)] for lbl in y_train])  # raw labels before padding

# input_lengths_test = np.ones((X_test.shape[0], 1)) * 44  # based on your reshape layer
# label_lengths_test = np.array([[len(lbl)] for lbl in y_test])  # raw labels before padding




# X_train = X_train[:50]
# y_train_enc= y_train_enc[:50]
# y_train = y_train[:50]

# X_test = X_test[:50]
# y_test_enc= y_test_enc[:50]
# y_test = y_test[:50]


#%%

# Assuming input_length is fixed
input_length = 44

# This will be your full dataset
max_allowed_label_length = input_length  # CTC requirement

# Filter samples where label length > input_length
valid_indices = [i for i, lbl in enumerate(y_train) if len(lbl) <= max_allowed_label_length]

# Apply filtering to all inputs
Xtrain = X_train[valid_indices]
y_train_enc = y_train_enc[valid_indices]
y_train = [y_train[i] for i in valid_indices]

# Recompute label lengths
input_lengths_train = np.full((len(y_train), 1), fill_value=input_length, dtype='int64')
label_lengths_train = np.array([[len(lbl)] for lbl in y_train], dtype='int32')
#%%

# Assuming input_length is fixed
input_length = 44

# This will be your full dataset
max_allowed_label_length = input_length  # CTC requirement

# Filter samples where label length > input_length
valid_indices = [i for i, lbl in enumerate(y_test) if len(lbl) <= max_allowed_label_length]

# Apply filtering to all inputs
Xtest = X_test[valid_indices]
y_test_enc = y_test_enc[valid_indices]
y_test = [y_test[i] for i in valid_indices]

# Recompute label lengths
input_lengths_test = np.full((len(y_test), 1), fill_value=input_length, dtype='int64')
label_lengths_test = np.array([[len(lbl)] for lbl in y_test], dtype='int32')
# %%

inputs_train = {
    'img_input': X_train,
    'ground_truth_labels': y_train_enc,
    'input_length': input_lengths_train,
    'label_length': label_lengths_train,
}

inputs_test = {
    'img_input': X_test,
    'ground_truth_labels': y_test_enc,
    'input_length': input_lengths_test,
    'label_length': label_lengths_test,
}



# inputs = {
#                 'img_input': X_train,  
#                 'ground_truth_labels': y_train_enc,  
#                 'input_length': input_length,  
#                 'label_length': label_length,
#                 # 'source_str': source_str  # used for visualization only
#             }
# inputs={X_train, y_train_enc, input_length, label_length}
dummy_targets_train = np.zeros((X_train.shape[0], 1))
dummy_targets_test = np.zeros((X_test.shape[0], 1))
img_text_recog.fit(x=inputs_train, y=dummy_targets_train,batch_size=32, epochs=5, validation_data=(inputs_test, dummy_targets_test), callbacks=[early_stop, tensorboard_callback,  model_chk_pt])

history = img_text_recog.fit()
img_text_recog.save('Img_recog_LSTM_Adam_model_run_3.h5')
#callbacks=[viz_cb_train,viz_cb_val,train_gene,val_gen,tensorboard_callback,early_stop,model_chk_pt],
# %%
import io
recog_model = model_definition('inference')
# recog_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
recog_model.load_weights('model.03-27.35.weights.h5')
# img_1 = X_test[0]
test_img_path = 'outputs/AA-GL 163.png'
with io.open(test_img_path, 'rb') as image_file:
    image_data = image_file.read()


img_proc, conf, inv = process_uploaded_image(image_data, None)
inv = inv[np.newaxis,..., np.newaxis]
# inv = inv.T
# inv = inv[..., ]
prediction = recog_model.predict(inv)
# plot_model(model_instance.model, show_shapes=True, show_layer_names=True)


# %%
import cv2
from PIL import Image
from IPython.display import Image as display
import matplotlib.pyplot as plt
test_img=np.array(Image.open(test_img_path))
test_img_resized=cv2.resize(test_img,(160,90))
test_image=test_img_resized[:,:,1]
# test_image=test_image.T
test_image=np.expand_dims(test_image,axis=-1)
test_image=np.expand_dims(test_image, axis=0)
test_image=test_image/255
prediction = recog_model.predict(test_image)
# Image(test_image)


# decode_label(prediction)
