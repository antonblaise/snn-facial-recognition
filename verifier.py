import os
os.system('cls' if os.name == 'nt' else 'clear')

print('\nFacial Recognition with Siamese Neural Network\n')

print('\n>> Starting...')

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tkinter import messagebox

print(' [DONE]\n')

print('\n>> Loading neural network model...\n')

class L1Dist(Layer): # from tensorflow.keras.layers import Layer

    # init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
    
    # Combine the two rivers
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

snn_model = tf.keras.models.load_model('snn_model', custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy}, compile=False)

print('\n [DONE]\n')

print('\n>> Starting engine...')

def preprocess(file_path):
    byte_img = tf.io.read_file(file_path) # read the file's data as bytes
    img = tf.io.decode_jpeg(byte_img) # decode the bytes as JPEG and store into img variable
    img = tf.image.resize(img, (100,100)) # resize the img into 100x100
    img /= 255.0 # Without this step the image will be super bright. Try it.
    return img

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

print(' [DONE]\n')

print('\n>> Starting camera...')

cap = cv2.VideoCapture(2)

while cap.isOpened():
    ret, frame = cap.read()

    frame = frame[130:250+130, 200:200+250, :]
    frame = cv2.putText(frame, '(Q)uit      (V)erify', (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 190, 240), 1)

    cv2.imshow('Verification', frame)

    # Verification trigger
    if cv2.waitKey(10) & 0xFF == ord('v'):

        messagebox.showwarning('Message', 'Verifying...\n(Click OK to continue)')

        # save the input image into the 'application_data\input_image' folder
        cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)

        # verification function
        results, verified = verify(snn_model, 0.7, 0.7)
        if verified:
            messagebox.showinfo('Success', 'Verification SUCCESSFUL!')
            # print('verification SUCCESS!')
        else:
            messagebox.showerror('Error', 'Verification FAILED')
            # print('verification FAILED')

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print('\n>> Camera closed.\n')