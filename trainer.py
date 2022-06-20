
#%% IMPORT DEPENDENCIES

# Import standard dependencies
from calendar import EPOCH
import cv2
import os
import os.path
import random
import numpy as np
from matplotlib import pyplot as plt

# Import Tensorflow dependencies
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Set GPU growth
# Avoid out-of-memory error by limiting GPU comsumption
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     print(gpu)
#     tf.config.experimental.set_memory_growth(gpu, True)

# Create folder structures
POS_PATH = os.path.join('data', 'positive') # ./data/positive
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# Create those folders (if they aren't created already)
if not os.path.exists(POS_PATH):
    os.makedirs(POS_PATH)
if not os.path.exists(NEG_PATH):
    os.makedirs(NEG_PATH)
if not os.path.exists(ANC_PATH):
    os.makedirs(ANC_PATH)

#%% COLLECT POSITIVES AND ANCHORS

# Move FaceID files into the following directory: data/negative
for directory in os.listdir('FaceID'):
    for file in os.listdir(os.path.join('FaceID', directory)):
        OLD_PATH = os.path.join('FaceID', directory, file)
        NEW_PATH = os.path.join(NEG_PATH, file)
        os.replace(OLD_PATH, NEW_PATH)

# Import uuid library to generate unique image names
import uuid

# Access webcam

cap = cv2.VideoCapture(2)

while cap.isOpened():
    ret, frame = cap.read() # ret = return value; frame = the actual image captured on webcam

    # slice/reshape our frame to size 250x250
    frame = frame[130:250+130, 200:200+250, :]

    # collect anchors when hit 'A'
    if cv2.waitKey(1) & 0xFF == ord('a'):
        imgName = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1())) # image name and full path to save
        cv2.imwrite(imgName, frame) # save image

    # Collect positives when hit 'P'
    if cv2.waitKey(1) & 0xFF == ord('p'):
        imgName = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1())) # image name and full path to save
        cv2.imwrite(imgName, frame) # save image

    cv2.imshow('Image Collection', frame) # render/show the captured image onto the screen

    # quit when hit 'Q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# disconnect webcam, close the window
cap.release()
cv2.destroyAllWindows()

#%% LOAD AND PREPROCESS IMAGES

# Get image directories

# Take 300 samples/images from each (must have matching number of samples)
anchor = tf.data.Dataset.list_files(f'{ANC_PATH}\*.jpg').take(300)
negative = tf.data.Dataset.list_files(f'{NEG_PATH}\*.jpg').take(300)
positive = tf.data.Dataset.list_files(f'{POS_PATH}\*.jpg').take(300)

# Preprocessing - Scale and Resize
def preprocess(file_path):
    byte_img = tf.io.read_file(file_path) # read the file's data as bytes
    img = tf.io.decode_jpeg(byte_img) # decode the bytes as JPEG and store into img variable
    img = tf.image.resize(img, (100,100)) # resize the img into 100x100
    img /= 255.0 # Without this step the image will be super bright. Try it.
    return img

# Create labelled dataset
# (anchor, positive) => 1,1,1,1,1
# (anchor, negative) => 0,0,0,0,0
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor))))) # All 1's, as anchor matches the positive
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor))))) # All 0's, as anchor matches the negative
data = positives.concatenate(negatives) # joining positive and negative together

showData = data.as_numpy_iterator()
example = showData.next()
showData.next()

# Build train-test partition

def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)

# Sample input tuple for the function: 
# (b'data\\anchor\\3c539b2a-a2fd-11ec-9a85-ace2d36277c6.jpg',   <- Innput
#  b'data\\positive\\d56b4449-a2fd-11ec-b5f5-ace2d36277c6.jpg', <- Comparison
#  1.0)                                                         <- Result

result = preprocess_twin(*example) # * = unpack the tuple, so we dont have to type each input arguments one by one ourselves

# result = (preprocessed input image, preprocessed comparison image, label - 1 as correct, 0 as wrong)
print(type(result))

f, axarr = plt.subplots(2,1)
axarr[0].imshow(result[0])
axarr[1].imshow(result[1])
print(result[2])

# Build dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)

# display training data individually/one by one

print(type(data))

ddd = data.as_numpy_iterator()
print(type(ddd))

dddd = ddd.next()

f, axarr = plt.subplots(2,1)

axarr[0].imshow(dddd[0])
axarr[1].imshow(dddd[1])
print(dddd[2])

# Training partition

train_data = data.take(round(len(data) * 0.7)) # Round off the 70% of the length of 'data'
train_data = train_data.batch(16) # Make batches of 16
train_data = train_data.prefetch(8) # Start preprocessing the next set of images

# Testing partition

test_data = data.skip(round(len(data) * 0.7)) # Avoid taking the train_data
test_data = test_data.take(round(len(data) * 0.3)) # Take the rest 30% of 'data'
test_data = test_data.batch(16) # Make batches of 16
test_data = test_data.prefetch(8) # Start preprocessing the next set of images

#%% MODEL ENGINEERING

# Build embedding layer

def make_embedding():
    inp = Input(shape=(100,100,3), name='input_image')

    # 1st block
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)

    # 2nd block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    # 3rd block
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)

    # Final embedding block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[inp], outputs=[d1], name='embedding') # Compile the model

embedding = make_embedding()
embedding.summary()

# build distance layer

class L1Dist(Layer): # from tensorflow.keras.layers import Layer

    # init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
    
    # Combine the two rivers
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

# Make Siamese model
def make_siamese_model():

    # Anchor image input
    input_image = Input(name='input_img', shape=(100, 100, 3))

    # Validation image (comparison)
    validation_image = Input(name='validation_img', shape=(100, 100, 3))

    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    # Classification layer (check - are they similar?)
    classifier = Dense(1, activation='sigmoid')(distances) # refer Model.png

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

snn_model = make_siamese_model()
snn_model.summary()

#%% TRAINING

# Setup loss and optimiser
# Define loss
binary_cross_loss = tf.losses.BinaryCrossentropy()

# Define optimiser
opt = tf.keras.optimizers.Adam(1e-4) # learning rate = 0.0004

# Establish checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=snn_model)

# Build train step function
@tf.function
def train_step(batch):

    # Record all of our operations
    with tf.GradientTape() as tape:
        # Get anchor and positive/negative image
        X = batch[:2]

        # Get label
        y = batch[2]

        # Forward pass
        yhat = snn_model(X, training=True)

        # Calculate loss
        loss = binary_cross_loss(y, yhat) # declared/defined under 'Setup loss and optimiser'

    print(loss)

    # Calculate gradients
    grad = tape.gradient(loss, snn_model.trainable_variables)

    # Calculate updated weights and apply to the siamese model
    opt.apply_gradients(zip(grad, snn_model.trainable_variables))

    return loss

# Build training loop
def train(data, EPOCHS): 
    # Loop through the epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))

        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            train_step(batch)
            progbar.update(idx+1)

        # Save checkpoints
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix) # Both 'checkpoint' and 'checkpoint_prefix' are defined under 'Establish checkpoints' above

EPOCHS = 50
train(train_data, EPOCHS)

#%% EVALUATE MODEL

# Import metric calculations
from tensorflow.keras.metrics import Precision, Recall

# Unpack the 3 components of test_data into individual parts
test_input, test_val, y_true = test_data.as_numpy_iterator().next()

# Make predictions
y_hat = snn_model.predict([test_input, test_val])

# Calculate the metrics
# 1. using Recall()
# Create a metric object
m = Recall()

# Calculating the recall value
m.update_state(y_true, y_hat)

# Return Recall Result 
print(f'Recall result = {m.result().numpy()}')

# 2. Using Precision()
# Create a metric object
m = Precision()

# Calculating the recall value
m.update_state(y_true, y_hat)

# Return Recall Result 
print(f'Precision result = {m.result().numpy()}')

# Visualise the results
plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(test_input[2]) # can take from index 0-15
plt.subplot(1,2,2)
plt.imshow(test_val[2])

#%% Save the model

# Save as two formats - .pb and .h5
snn_model.save('smm_model')
snn_model.save('smm_model.h5')

# Load the saved model
model = tf.keras.models.load_model('smm_model', custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

# Make prediction using the loaded model
print(f'Prediction with loaded model:\n{model.predict([test_input, test_val])}')

#%% REAL TIME TEST

def verify(model, detection_threshold, verification_threshold):
    results = []
    for image in os.listdir(os.path.join('application_data','verification_images')): # loop through every image in the 'verification_images' folder
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg')) # snap one image from the webcam as save it as 'input_image.jpg' inside that directory
        verification_img = preprocess(os.path.join('application_data', 'verification_images', image))
        results.append(model.predict(list(np.expand_dims([input_img, verification_img], axis=1)))) # compare the input image with the verification images
        
    # Sum up all the results that exceeds the detection threshold
    detection = np.sum(np.array(results) > detection_threshold)

    # Proportion of verification
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
    
    # if proportion of verification > verification threshold, then verified = True
    verified = verification > verification_threshold

    return results, verified







