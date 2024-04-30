from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import *
import os
import numpy as np
import neurokit2 as nk
from scipy import signal
from tqdm import tqdm
from scipy.io import loadmat
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from scipy.io import loadmat
from keras.preprocessing.sequence import pad_sequences
import lime
from lime import lime_tabular
import os
# 데이터 읽기

def load_challenge_data(filename):
    x = loadmat(filename)
    # print("load_challenge_data_x : ", x)
    data = np.asarray(x['val'], dtype=np.float64)
    # print("*"*100)
    # print("load_challenge_data_data : ", data)
    # print("*"*100)
    # print("load_challenge_data_data.shape : ", data.shape)
    # print("*"*100)
    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file,'r') as f:
        header_data=f.readlines()
    # print("load_challenge_data_header_data : ", header_data)
    # print("*"*100)
    return data, header_data


def import_ecg_data(directory, ecg_len = 5000, trunc="post", pad="post"):
    print("Starting ECG import..")
    ecgs = []
    for ecgfilename in tqdm(sorted(os.listdir(directory))):
        filepath = directory + os.sep + ecgfilename
        if filepath.endswith(".mat"):
            data, header_data = load_challenge_data(filepath)
            data = pad_sequences(data, maxlen=ecg_len, truncating=trunc,padding=pad)
            # print("data : ", data)
            ecgs.append(data)
    print("Finished!")
    return np.asarray(ecgs)

def resample_beats(beats):
    rsmp_beats=[]
    for i in beats:
        i = np.asarray(i)

        #i = i[~np.isnan(i)]
        f = signal.resample(i, 250)
        rsmp_beats.append(f)
    rsmp_beats = np.asarray(rsmp_beats)
    return rsmp_beats

def median_beat(beat_dict):
    beats = []
    for i in beat_dict.values():
        #print(i['Signal'])
        beats.append(i['Signal'])
    beats = np.asarray(beats)
    rsmp_beats = resample_beats(beats)
    med_beat = np.median(rsmp_beats,axis=0)
    return med_beat

def process_ecgs(raw_ecg):    
    processed_ecgs=[]
    for i in tqdm(range(len(raw_ecg))):
        leadII = raw_ecg[i][1]
        leadII_clean = nk.ecg_clean(leadII, sampling_rate=500, method="neurokit")
        r_peaks = nk.ecg_findpeaks(leadII_clean, sampling_rate=500, method="neurokit", show=False)
        twelve_leads = []
        for j in raw_ecg[i]:
            try:
                beats = nk.ecg_segment(j, rpeaks=r_peaks['ECG_R_Peaks'], sampling_rate=500, show=False)
                med_beat = median_beat(beats)
                twelve_leads.append(med_beat)
            except:
                beats = np.ones(250)*np.nan
                twelve_leads.append(beats)
        #twelve_leads = np.asarray(twelve_leads)
        processed_ecgs.append(twelve_leads)
    processed_ecgs = np.asarray(processed_ecgs)
    return processed_ecgs
    
def remove_nans(ecg_arr):
    new_arr = []
    for i in tqdm(ecg_arr):
        twelve_lead = []  
        for j in i:
            if j[0] != j[0]:
                j = np.ones(250)
            twelve_lead.append(j)
        new_arr.append(twelve_lead)
    new_arr = np.asarray(new_arr)
    return new_arr

def remove_some_ecgs(ecg_arr):
    delete_list = []
    for i in tqdm(range(len(ecg_arr))):
        if np.all(ecg_arr[i].T[0]==1):
            delete_list.append(i)
    ecg_arr = np.delete(ecg_arr,delete_list,axis=0)
    return ecg_arr

def average_and_rebin(array, bin_size):
    new_arr =[]
    temp_len = int(len(array)/bin_size)
    for i in range(temp_len):
        bin = []
        for j in range(bin_size):
            bin_val = array[j + (i*bin_size)]
            bin.append(bin_val)
        new_arr.append(np.repeat(np.mean(bin),bin_size))
    return np.asarray(new_arr).ravel()
normal_sinus_dir = "../dataset/only_sinus"
wpw_dir = "../dataset/only_sinus"

# ecg 데이터 읽어오기
normal_sinus = import_ecg_data(normal_sinus_dir)
print("normal_sinus.ndim",normal_sinus.ndim)
print("normal_sinus.shape",normal_sinus.shape)
wpw = import_ecg_data(wpw_dir)

#For test purpose we chose to only use a fraction of the total ECGs
norm_ecgs = process_ecgs(normal_sinus[:100,:,:])
print("norm_ecgs.ndim : ",norm_ecgs.ndim)
print("norm_ecgs.shape : ",normal_sinus.shape)

#For test purpose we chose to only use a fraction of the total ECGs
wpw_ecgs = process_ecgs(wpw[:100,:,:])



# 결측치 제거
new_norm = remove_nans(norm_ecgs)
new_wpw = remove_nans(wpw_ecgs)

# 오류있는 ecg 데이터 삭제
clean_wpw = remove_some_ecgs(new_wpw)
clean_norm = remove_some_ecgs(new_norm)

# Reshape the ECG arrays
print("clean_wpw :",clean_wpw.shape)
print("clean_norm :",clean_norm.shape)
clean_wpw = np.moveaxis(clean_wpw, 1, -1)
clean_norm = np.moveaxis(clean_norm, 1, -1)
print("Reshapeclean_wpw :",clean_wpw.shape)
print("Reshapeclean_norm :",clean_norm.shape)

# training split normal ECG
norm_train = clean_norm[:-30]
norm_val = clean_norm[-30:]
print("norm_train : ",norm_train,norm_train.shape)
print("norm_val : ",norm_val,norm_val.shape)
# training split WPW ECG
wpw_train = clean_wpw[:50]
wpw_val = clean_wpw[50:]

y_norm_train = np.zeros(norm_train.shape[0])
y_norm_val = np.zeros(norm_val.shape[0])
y_wpw_train = np.ones(wpw_train.shape[0])
y_wpw_val = np.ones(wpw_val.shape[0])

print(y_norm_train)

print(y_wpw_train)

X_train = np.vstack([norm_train,wpw_train])
y_train = np.hstack([y_norm_train,y_wpw_train])

X_val = np.vstack([norm_val,wpw_val])
y_val = np.hstack([y_norm_val,y_wpw_val])
def grad_cam(layer_name, data, model):
    grad_model = tf.keras.models.Model(
    [model.inputs], [model.get_layer(layer_name).output, model.output]
)
    last_conv_layer_output, preds = grad_model(data)

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(data)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)


    pooled_grads = tf.reduce_mean(grads, axis=(0))

    last_conv_layer_output = last_conv_layer_output[0]

    heatmap = last_conv_layer_output * pooled_grads
    heatmap = tf.reduce_mean(heatmap, axis=(1))
    heatmap = np.expand_dims(heatmap,0)
    return heatmap
def FCN():
    inputlayer = keras.layers.Input(shape=(250,12), name="first_layer") 

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=15,input_shape=(250,12), padding='same', name="first_conv")(inputlayer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)
    conv1 = keras.layers.SpatialDropout1D(0.1)(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=10, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)
    conv2 = keras.layers.SpatialDropout1D(0.1)(conv2)

    conv3 = keras.layers.Conv1D(512, kernel_size=5,padding='same', name="last_conv")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)
    conv3 = keras.layers.Dropout(0.2)(conv3)

    gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = tf.keras.layers.Dense(units=2,activation='softmax')(gap_layer)

    model = keras.Model(inputs=inputlayer, outputs=output_layer)
    
    
    return model


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.1, patience=1, verbose=1, mode='max',
    min_delta=0.0001, cooldown=5, min_lr=0
)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=3, patience=15, restore_best_weights=True)

class_weights = np.ones(shape=(len(y_train),))
class_weights[y_train == 1] = len(norm_train)/len(wpw_train)



model = FCN()
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=["accuracy"])
batchsize=30
history = model.fit(x=X_train,y=tf.keras.utils.to_categorical(y_train), epochs=50, batch_size=batchsize, 
        validation_data=(X_val,tf.keras.utils.to_categorical(y_val)),steps_per_epoch=(len(X_train)/batchsize), 
        shuffle=True, sample_weight=class_weights, callbacks=[reduce_lr,early_stop])
model.save('wpw_model.h5')
#공유
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("./acc.png")

# 7 훈련 과정 시각화 (손실)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("./loss.png")

explainer = lime_tabular.RecurrentTabularExplainer(X_val[:-6],training_labels=tf.keras.utils.to_categorical(y_val[:-10]), feature_names=["Lead-I","Lead-II", "Lead-III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"],
                                                discretize_continuous=False, feature_selection='auto', class_names=['Normal','WPW'])
for i,j in enumerate(X_val[:4]):
    print("predicted labels ([Normal, WPW])")
    print(model.predict(np.expand_dims(j,axis=0))[0])
    print("-----------------------")
    print("WPW = 1, Normal = 0")
    print("correct label is: {}".format(y_val[-1*(i+1)]))
    print("-----------------------")
    exp = explainer.explain_instance(np.expand_dims(j,axis=0),model.predict, num_features=250*12)
    explanations = exp.as_list()
    heatmap = np.zeros([12,250])

    for k in explanations:
        if k[0].split("_")[0]=='Lead-I':
            heatmap[0][int(k[0].split("-")[-1])]=k[1]
        elif k[0].split("_")[0]=='Lead-II':
            heatmap[1][int(k[0].split("-")[-1])]=k[1]
        elif k[0].split("_")[0]=='Lead-III':
            heatmap[2][int(k[0].split("-")[-1])]=k[1]
        elif k[0].split("_")[0]=='aVR':
            heatmap[3][int(k[0].split("-")[-1])]=k[1]
        elif k[0].split("_")[0]=='aVL':
            heatmap[4][int(k[0].split("-")[-1])]=k[1]
        elif k[0].split("_")[0]=='aVF':
            heatmap[5][int(k[0].split("-")[-1])]=k[1]
        elif k[0].split("_")[0]=='V1':
            heatmap[6][int(k[0].split("-")[-1])]=k[1]
        elif k[0].split("_")[0]=='V2':
            heatmap[7][int(k[0].split("-")[-1])]=k[1]
        elif k[0].split("_")[0]=='V3':
            heatmap[8][int(k[0].split("-")[-1])]=k[1]
        elif k[0].split("_")[0]=='V4':
            heatmap[9][int(k[0].split("-")[-1])]=k[1]
        elif k[0].split("_")[0]=='V5':
            heatmap[10][int(k[0].split("-")[-1])]=k[1]
        elif k[0].split("_")[0]=='V6':
            heatmap[11][int(k[0].split("-")[-1])]=k[1]
            
    test_heatmap = heatmap.copy()
    test_heatmap[np.where(heatmap > 0)] = 0.0
    test_heatmap = abs(test_heatmap)
    leads = ["Lead-I","Lead-II", "Lead-III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
    plt.figure(figsize=(26, 16))
    for l in range(12):
        plt.subplot(3, 4, l + 1)
        plt.title("Explain WPW - Lead: {} ".format(leads[l]), fontsize=20)

        plt.imshow(np.expand_dims(average_and_rebin(test_heatmap[l],7),axis=0),cmap='Reds', aspect="auto", interpolation='nearest',extent=[0,250,round(j[:,l].min()*1.05),round(j[:,l].max()*1.05)],
                vmin=test_heatmap.min(), vmax=test_heatmap.max(), alpha=1.0)
        plt.plot(j[:,l],'k')

    #plt.colorbar()
    plt.savefig("wpw_{}.png".format(i),dpi=300)
    plt.suptitle("ECG_{}".format(i))
    plt.show()  
for example in X_val[:10]:
    exp = grad_cam("last_conv", np.expand_dims(example,0), model)
    plt.imshow(exp,cmap='Reds', aspect="auto", interpolation='nearest',extent=[0,250,round(example.min()*1.05),round(example.max()*1.05)],
        vmin=exp.min(), vmax=exp.max(), alpha=1.0)
    plt.plot(example[:,1],'k')
    plt.show()


# class MyModel(tf.keras.Model):

#   def __init__(self):
#     super().__init__()
#     self.Conv1d = Sequential(
#                Conv1D(filters=128, kernel_size=15,input_shape=(250,12), padding='same', name="first_conv"),
#                BatchNormalization(),
#                Activation(activation='relu'),
#                SpatialDropout1D(0.1),

#                Conv1D(filters=256, kernel_size=10, padding='same'),
#                BatchNormalization(),
#                Activation(activation='relu'),
#                SpatialDropout1D(0.1),

#                Conv1D(filters=512, kernel_size=5, padding='same'),
#                BatchNormalization(),
#                Activation(activation='relu'),
#                SpatialDropout1D(0.1),
#     )
#     self.lstm = Sequential(
#      LSTM(128, input_shape=(250,12)),
#      LSTM(256),
#      LSTM(512),

#     )

#   def call(self, inputs):
#     x1 = self.Conv1d(inputs)
#     x2 = self.lstm(inputs)
#     x = concatenate([x1, x2])
#     x = Dense(x1.shape[1] + x2.shape[1], activation='relu')(x)
#     main_output = Dense(2, activation='sigmoid', name='main_output')(x)
#     return main_output

# model_path = r"./"
# model = MyModel()
# model.compile(loss='mean_squared_error', optimizer='adam')
# early_stop = EarlyStopping(monitor='val_loss', patience=5)
# filename = os.path.join(model_path, 'tmp_checkpoint.h5')
# checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

# history = model.fit(X_train, y_train, 
#                     epochs=200, 
#                     batch_size=16,
#                     validation_data=(X_val, y_val), 
#                     callbacks=[early_stop, checkpoint])