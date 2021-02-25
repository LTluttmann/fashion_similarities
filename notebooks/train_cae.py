# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] id="BiBbe4RSmCvK"
# This notebook can be used to train the convolutional autoencoder from the fashion-similarities package on a GPU-enable google colab notebook. The .whl of the package has to be placed on google drive and the path to it must be specified. 

# + id="8VTG7T83oung" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1614241127539, "user_tz": -60, "elapsed": 695993, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="89398e2d-4047-41db-e75c-6cf8c02b684d"
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

# + colab={"base_uri": "https://localhost:8080/"} id="L49SqQ5fAHha" executionInfo={"status": "ok", "timestamp": 1614241525744, "user_tz": -60, "elapsed": 1094180, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="f7bcd528-8f1e-42ff-cbdc-2e367a6c892d"
# %run /content/gdrive/MyDrive/colab_notebooks/deep_learning/load_and_split_data.ipynb

# + id="2IEbAw4ppBTM" executionInfo={"status": "ok", "timestamp": 1614241526118, "user_tz": -60, "elapsed": 1094551, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
# standard modules
import os
import matplotlib.gridspec as gridspec
import joblib
import time

# deep learning modules
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau

# other data science modules
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import pyplot

# fashion similarities package
from fashion_similarities.lsh import LSH
from fashion_similarities.utils import GetSimpleData
from fashion_similarities.autoencoder import ConvAutoencoder

# + colab={"base_uri": "https://localhost:8080/"} id="Oo340zQv6aNB" executionInfo={"status": "ok", "timestamp": 1614241532496, "user_tz": -60, "elapsed": 1100910, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="f8668d2a-6cd8-48f5-a346-05bd8598186e"
# see if model is imported correctly
ae = ConvAutoencoder.build(224, 224, 3)
ae.summary()

# + colab={"base_uri": "https://localhost:8080/", "height": 612} id="YV85idAR5I0w" executionInfo={"status": "ok", "timestamp": 1614241533865, "user_tz": -60, "elapsed": 1102262, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="6e1fdea8-f8aa-4dcb-beb4-4cc0980c9559"
# look at data
rows = 5
cols = 3
plt.figure(figsize=(18, 10))
gs1 = gridspec.GridSpec(3, 5)
gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 

np.random.seed(415)
for r in range(rows):
  for c in range(cols):
    # display original
    #ax = plt.subplot(3, rows, c*rows + r + 1)

    ax = plt.subplot(gs1[c*rows + r])
    i = np.random.random_integers(0, len(images)-1)
    img = load_img(images[i])  # this is a PIL image
    plt.imshow(img)
    #plt.title("original")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.subplots_adjust(wspace=None, hspace=None)
plt.show()

# + id="pSy-6gcX3zAI" executionInfo={"status": "ok", "timestamp": 1614241534078, "user_tz": -60, "elapsed": 1102471, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
train_img = GetSimpleData.list_files("img_train")
tets_img = GetSimpleData.list_files("img_test")
val_img = GetSimpleData.list_files("img_validation")

# + id="pVkvQNZb1ohf" executionInfo={"status": "ok", "timestamp": 1614241534079, "user_tz": -60, "elapsed": 1102469, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
image_height, image_width, channels = img_to_array(load_img(train_img[0])).shape

# + colab={"base_uri": "https://localhost:8080/", "height": 595} id="0kMRyeiZwTij" executionInfo={"status": "ok", "timestamp": 1614241536452, "user_tz": -60, "elapsed": 1104821, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="9f781f54-81a2-43af-cb32-09f674826377"
datagen = ImageDataGenerator(
    #rotation_range=40,
    rescale=1./255,
    #zoom_range=[.8,1],
    #fill_mode='nearest',
    #shear_range=0.8,
    zoom_range=0.25,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest"
)  
# prepare iterator
exmpl_img = np.expand_dims(img_to_array(load_img(train_img[1])), 0)
it = datagen.flow(exmpl_img, batch_size=1)
# generate samples and plot
n = 25
plt.figure(figsize=(10,10))
for i in range(n):
	# define subplot
	pyplot.subplot(5, 5, i + 1)
	# generate batch of images
	batch = it.next()
	# convert to unsigned integers for viewing
	image = batch[0]
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()

# + id="WYk8HJy3JP3Q" executionInfo={"status": "ok", "timestamp": 1614241536456, "user_tz": -60, "elapsed": 1104820, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
# create a data generator
train_datagen = ImageDataGenerator(
    rescale=1/255,
    horizontal_flip=True
)

# + id="c4H7fJ4L5X6Q" executionInfo={"status": "ok", "timestamp": 1614241536457, "user_tz": -60, "elapsed": 1104818, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
test_datagen = ImageDataGenerator(
    rescale=1./255
)

# + id="_LkxQ04x5SW2" executionInfo={"status": "ok", "timestamp": 1614241536458, "user_tz": -60, "elapsed": 1104817, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
val_datagen = ImageDataGenerator(
    rescale=1/255
)

# + id="TfXPuxqiTA12" executionInfo={"status": "ok", "timestamp": 1614241536460, "user_tz": -60, "elapsed": 1104816, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
EPOCHS = 50
INIT_LR = 1e-2
BS = 32

# + colab={"base_uri": "https://localhost:8080/"} id="JUjsD_hoJP50" executionInfo={"status": "ok", "timestamp": 1614241536821, "user_tz": -60, "elapsed": 1105166, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="6dda8e6c-a074-4833-d9a4-7f98cc1660e4"
kwargs = dict(class_mode='input', target_size=(image_width, image_height), color_mode="rgb", batch_size=BS, shuffle=True)
# load and iterate training dataset
train_it = train_datagen.flow_from_directory('.', classes=["img_train"], **kwargs)
# load and iterate validation dataset
val_it = val_datagen.flow_from_directory('.', classes=["img_validation"], **kwargs)
# load and iterate test dataset
test_it = test_datagen.flow_from_directory('.', classes=["img_test"], **kwargs)

# + id="EBSpEz8akrTC" executionInfo={"status": "ok", "timestamp": 1614241536822, "user_tz": -60, "elapsed": 1105164, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
STEP_SIZE_TRAIN = train_it.n // train_it.batch_size
STEP_SIZE_VALID = val_it.n // val_it.batch_size

# + id="b6gwtOD9MNZT" executionInfo={"status": "ok", "timestamp": 1614241536823, "user_tz": -60, "elapsed": 1105163, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
AUTOENCODER_DIR = "/content/gdrive/MyDrive/conv_autoencoder"
autoencoder_checkpoint = ModelCheckpoint(
    AUTOENCODER_DIR, 
    monitor="val_loss", 
    verbose=1, save_best_only=True, 
    save_weights_only=False,
    save_format='tf'
)

# + id="nWw1Vr2lGfWq" executionInfo={"status": "ok", "timestamp": 1614241536823, "user_tz": -60, "elapsed": 1105159, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
RETRAIN = False

# + id="U1JKYfwlGPI1" executionInfo={"status": "ok", "timestamp": 1614241549980, "user_tz": -60, "elapsed": 1118281, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
if not os.path.exists(AUTOENCODER_DIR) or RETRAIN:
    print("goes here")
    autoencoder = ConvAutoencoder.build(
        image_width, 
        image_height, 
        channels
    )
    opt = Adam(lr=INIT_LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    autoencoder.compile(loss="mse", optimizer=opt)
    initial_epoch = 0
else:
    autoencoder = load_model(AUTOENCODER_DIR) #load the model from file
    initial_epoch=45
    INIT_LR = 1e-5
    epochs=50

# + colab={"base_uri": "https://localhost:8080/"} id="9Eqs1bM-fnu-" executionInfo={"status": "ok", "timestamp": 1614241549983, "user_tz": -60, "elapsed": 1118232, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="0f56c3c7-94e6-4b1d-b3f1-9fa2817c260a"
autoencoder.summary()

# + id="olptkxK3ncEZ" executionInfo={"status": "ok", "timestamp": 1614241549984, "user_tz": -60, "elapsed": 1118222, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
# ! if ! [ -d "$AUTOENCODER_DIR" ]; \
# then ( \
#     echo $AUTOENCODER_DIR ; \
#     mkdir $AUTOENCODER_DIR ; \
# ) ; \
# fi

# + id="4AQxj9TknS21" executionInfo={"status": "ok", "timestamp": 1614241549985, "user_tz": -60, "elapsed": 1118217, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-5)

# + colab={"base_uri": "https://localhost:8080/"} id="fvfM7iI3SwtG" executionInfo={"status": "ok", "timestamp": 1614244447964, "user_tz": -60, "elapsed": 4016088, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="b416fedb-513e-4912-ee9f-b197e4c2fffb"
history = autoencoder.fit(
    train_it,
    epochs=EPOCHS,
    initial_epoch=initial_epoch,
    steps_per_epoch=STEP_SIZE_TRAIN, 
    validation_data=val_it,
    validation_steps=STEP_SIZE_VALID,
    callbacks=[autoencoder_checkpoint]
)

# + id="KLdk8sq-jOTt"
joblib.dump(history, os.path.join(AUTOENCODER_DIR, "history.pkl"))

# + id="8CLSL9EKDARv"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# + id="gcLzUVmDZczU"
ENCODER_DIR = "/content/gdrive/MyDrive/conv_encoder"
encoder = load_model(ENCODER_DIR)

# + [markdown] id="mjWVZ5YR9wKi"
# # Analyse Results

# + id="OpZfo03V2HA_" executionInfo={"status": "ok", "timestamp": 1614244530459, "user_tz": -60, "elapsed": 7930, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
autoencoder = load_model(AUTOENCODER_DIR)

# + id="W89hxrWxcgMK" colab={"base_uri": "https://localhost:8080/", "height": 344} executionInfo={"status": "ok", "timestamp": 1614244584468, "user_tz": -60, "elapsed": 1895, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="d81730ad-c0b3-4f8a-de1d-6145c53a83b5"
n = 5
plt.figure(figsize=(12, 6))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  test_x = next(test_it)[0][0]
  plt.imshow(test_x)
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  # plt.imshow(decoded_imgs[i])
  plt.imshow(autoencoder.predict(test_x.reshape(1,image_height,image_width,channels))[0])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()

# + id="FO2SVoVWC2mR" colab={"base_uri": "https://localhost:8080/", "height": 286} executionInfo={"status": "ok", "timestamp": 1614244532530, "user_tz": -60, "elapsed": 7649, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="9cdad1f3-827a-427f-e206-2ae3d434730e"
plt.imshow(autoencoder.predict(test_x.reshape(1,image_height,image_width,channels))[0])

# + id="s17e3IzPC3q-" colab={"base_uri": "https://localhost:8080/", "height": 286} executionInfo={"status": "ok", "timestamp": 1614244532533, "user_tz": -60, "elapsed": 7463, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="3d981d68-9877-4ca2-d8b0-4441d5f0b4c7"
plt.imshow(test_x)

# + id="bV0UwMcplbE4" executionInfo={"status": "ok", "timestamp": 1614244532534, "user_tz": -60, "elapsed": 7267, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
test_x = next(test_it)

# + id="VMkl_mHJY_BK" colab={"base_uri": "https://localhost:8080/", "height": 286} executionInfo={"status": "ok", "timestamp": 1614244533303, "user_tz": -60, "elapsed": 7860, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="5c6b82d5-0073-4f5e-a5ea-6dea8f38ffe0"
plt.imshow(autoencoder.predict(test_x[0][0].reshape(1,image_height,image_width,channels))[0])

# + id="SBcsSNMqdJld" colab={"base_uri": "https://localhost:8080/", "height": 286} executionInfo={"status": "ok", "timestamp": 1614244533305, "user_tz": -60, "elapsed": 7687, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="42aa8f39-6d39-4277-ba4a-688975916a11"
plt.imshow(test_x[0][0])

# + id="GPfFlp4fm7Lw" executionInfo={"status": "ok", "timestamp": 1614244606006, "user_tz": -60, "elapsed": 961, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
test_img = GetSimpleData.list_files("img_test")
test_img = test_img[:2500]

# + id="1dgidpSum3Sn" executionInfo={"status": "ok", "timestamp": 1614244608623, "user_tz": -60, "elapsed": 3416, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
test_x = np.ndarray(shape=(len(test_img), image_height, image_width, channels),
                     dtype=np.float32)

for i, _file in enumerate(test_img):
    img = load_img(_file)  # this is a PIL image
    img.thumbnail((image_height, image_width))
    # Convert to Numpy Array
    x = img_to_array(img) 
    x = x.reshape((image_height, image_width, channels))
    test_x[i] = x


# + id="5kq6qqIKN_0C" executionInfo={"status": "ok", "timestamp": 1614244614588, "user_tz": -60, "elapsed": 9205, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
layer_name = 'encoded'
intermediate_layer_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(test_x)

# + id="lIHRgalx0o2P" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1614244614595, "user_tz": -60, "elapsed": 9051, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="e3c06294-18c1-4db5-c14f-41d273538e1c"
intermediate_output

# + id="U7BUzvQnnvsu" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1614244614597, "user_tz": -60, "elapsed": 8856, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="7e9af4dc-5aa8-42b9-8650-5b29f494898b"
intermediate_layer_model.output.shape[-1]

# + id="yQf0rmqim3Xl" executionInfo={"status": "ok", "timestamp": 1614244645641, "user_tz": -60, "elapsed": 810, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
query_idx=2
a = intermediate_output[query_idx]
b = {i: intermediate_output[i] for i in [x for x in range(0, len(intermediate_output)) if x != query_idx]}
dists = {idx: LSH.calc_dist_euclidean(a,embedding) for idx, embedding in b.items()}
min_index = min(dists, key=dists.get)

# + id="FHyP8CnyorHA" colab={"base_uri": "https://localhost:8080/", "height": 286} executionInfo={"status": "ok", "timestamp": 1614244646331, "user_tz": -60, "elapsed": 1305, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="9d8aef26-b778-42ea-fcaa-c82a28d11136"
plt.imshow(test_x[query_idx].astype("int"))

# + id="7qeqgqHM_eO6" colab={"base_uri": "https://localhost:8080/", "height": 286} executionInfo={"status": "ok", "timestamp": 1614244646334, "user_tz": -60, "elapsed": 1101, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="5625814f-94e4-44a0-ec3d-e1ac65e3423b"
plt.imshow(test_x[min_index].astype("int"))

# + [markdown] id="JbYfqDyJ335X"
# ## Analyse Embedding

# + id="PpVgAm9rzq3a" executionInfo={"status": "ok", "timestamp": 1614244652579, "user_tz": -60, "elapsed": 588, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
layer_name = 'encoded'
intermediate_layer_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(layer_name).output)


# + id="5GJH-dMBzwcq" executionInfo={"status": "ok", "timestamp": 1614244652938, "user_tz": -60, "elapsed": 744, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
def get_embedding(encoder, image_data):
    image_data = np.expand_dims(image_data, 0) if not len(image_data.shape) == 4 else image_data
    image_data = image_data.astype('float32') / 255.
    encoded_img = encoder.predict(image_data)
    return encoded_img


# + id="VskF1g6rzwfc" executionInfo={"status": "ok", "timestamp": 1614244653194, "user_tz": -60, "elapsed": 803, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
tst_images = GetSimpleData.list_files("img_test")

# + id="tlWiTLeO8d06" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1614244654203, "user_tz": -60, "elapsed": 1632, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="70994fe2-1522-44e9-9eeb-e12a9221520a"
meta_df = pd.read_csv("/content/gdrive/MyDrive/styles.csv", error_bad_lines=False)

# + id="TXPIlQlQ8eMa" executionInfo={"status": "ok", "timestamp": 1614244654206, "user_tz": -60, "elapsed": 1442, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
df_img = pd.DataFrame.from_dict(
    {int(tst_img.split("_")[-1].split(".")[0]): tst_img for tst_img in tst_images}, orient="index", columns=["image"]
)
df = meta_df.merge(df_img, left_on="id", right_index=True)

# + id="azMzbkUZ4wdV" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1614244654209, "user_tz": -60, "elapsed": 1281, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="3c046da5-900e-471a-e198-aeaee8040741"
df.shape

# + id="sFMAe-pk7H2u" executionInfo={"status": "ok", "timestamp": 1614244816699, "user_tz": -60, "elapsed": 163602, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
preds = {}
for i, row in df.iterrows():
    img = img_to_array(load_img(row.image))
    emb = get_embedding(intermediate_layer_model, img)
    preds[i] = emb

# + id="zhVxjAYZ8Rhx" executionInfo={"status": "ok", "timestamp": 1614244816706, "user_tz": -60, "elapsed": 163429, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
preds = {key:value[0] for key, value in preds.items()}

# + id="5bHRxmM48M8J" executionInfo={"status": "ok", "timestamp": 1614244818310, "user_tz": -60, "elapsed": 164872, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
df_emb = pd.DataFrame.from_dict(preds, orient="index")

# + id="F8BFh3RZ3Rwd" colab={"base_uri": "https://localhost:8080/", "height": 156} executionInfo={"status": "ok", "timestamp": 1614244818316, "user_tz": -60, "elapsed": 164708, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="d996acc9-abf7-41a6-ff39-eac7f2269b11"
df_emb.head(2)

# + id="6V5Hoam2-r03" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1614244875540, "user_tz": -60, "elapsed": 221757, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="07fa1d84-9902-4413-cda3-cf01989fdeaa"
time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df_emb)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

# + id="CESNoB2T8wtl" colab={"base_uri": "https://localhost:8080/", "height": 107} executionInfo={"status": "ok", "timestamp": 1614244875546, "user_tz": -60, "elapsed": 221512, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="ea419dbb-057b-4c19-a49a-a1c32176e8c0"
df.head(2)

# + id="57nhpPE_-yct" executionInfo={"status": "ok", "timestamp": 1614244875549, "user_tz": -60, "elapsed": 221442, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}}
df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]

# + id="mqERoWKr6xTF" colab={"base_uri": "https://localhost:8080/", "height": 641} executionInfo={"status": "ok", "timestamp": 1614244877536, "user_tz": -60, "elapsed": 223191, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="8447821e-0bb2-4788-dec2-a7ba7a277620"
plt.figure(figsize=(16,10))
sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two",
                hue="subCategory",
                data=df,
                legend="full",
                alpha=0.8)
