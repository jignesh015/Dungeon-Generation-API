import uvicorn
import json
import random
from keras import models
import numpy as np 
from scipy import ndimage
from fastapi import FastAPI, Form
import helper
import main
import tensorflow as tf


def generate_dungeon(num_of_samples, model_choice):
    if model_choice == 0:  #DCGAN
        # Generate new data
        dcgan_input_noise = tf.random.normal([1, 100])
        generated_data = main.dungeon_dcgan(dcgan_input_noise, training=False)
        # Convert the range [-1, 1] to [0, 1]
        generated_data = generated_data.numpy()
        generated_data = (generated_data >= 0.5).astype(np.int32)
        return generated_data
    elif model_choice == 1:   #Autoencoder
        # Generate a sample dataset to pass to the model
        sample_arr = np.random.randint(low=0, high=2, size=(num_of_samples, 64))
        #Pass sample to the model
        predictions = main.dungeon_autoencoder.predict(sample_arr)
        return predictions
    
def generate_rooms(num_of_samples, model_choice):
    if model_choice == 0:  #DCGAN
        # Initialize an array to store the output from the model
        generated_levels = np.zeros((num_of_samples, 16, 16))

        for i in range(num_of_samples):
            # Generate new data
            dcgan_input_noise = tf.random.normal([1, 100])
            sample_data = main.rooms_dcgan(dcgan_input_noise, training=False)
            # Convert the range [0, 1] to [0, 8]
            sample_data = sample_data*8.0
            sample_data = np.round(np.clip(sample_data,0,8)).astype(int)
            sample_data = sample_data.reshape(sample_data.shape[0],16,16)
            generated_levels[i] = sample_data
            print("generated_levels",generated_levels.shape)
        return generated_levels
    elif model_choice == 1:   #Autoencoder
        # Initialize an array to store the output from the model
        generated_levels = np.zeros((num_of_samples, 16, 16))

        for i in range(num_of_samples):
            # Generate a sample dataset to pass to the model
            sample_data = np.random.randint(low=0, high=9, size=(1, 16,16,1)) / 8.0

            #Pass the sample data to the model
            level = main.rooms_cnn.predict(sample_data,verbose=0)
            level = np.squeeze(level)
            level = np.clip(level, 0, 8)  # clip to 0-8
            generated_levels[i] = level

        # Scale the generated levels to 0-8
        max_value = np.max(generated_levels)
        if max_value != 0:
            generated_levels *= 8.0/max_value
        return generated_levels
    