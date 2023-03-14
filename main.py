import uvicorn
import json
import random
from keras import models
import numpy as np 
from scipy import ndimage
from fastapi import FastAPI, Form
import helper
import generate
import tensorflow as tf

app = FastAPI()

# Load the datasets
dungeon_data = np.load('datasets\dungeons_dataset.npy')
room_data = np.load('datasets\\rooms_dataset.npy')

# Replace the invalid values with 1
dungeon_data[np.logical_and(dungeon_data != 0, dungeon_data != 1)] = 1

# Load the dungeon models
dungeon_dcgan = models.load_model('models\dungeon_dcgan.h5', compile=False)
dungeon_vae_model = models.load_model('models\dungeon_vae_model.h5', compile=False)
dungeon_autoencoder = models.load_model('models\dungeon_autoencoder.h5')

# Load the room models
rooms_dcgan =  models.load_model('models\\rooms_dcgan.h5', compile=False)
rooms_vae_model =  models.load_model('models\\rooms_vae_model.h5', compile=False)
rooms_cnn =  models.load_model('models\\rooms_cnn_model.h5')

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/generate_dungeons")
async def generate_dungeons(num_of_samples: int = Form(...), model_choice: int = Form(...), use_corrective_algorithm: int = Form(...)):
    #Generate appropriate predictions as per the given parameters
    predictions = generate.generate_dungeon(num_of_samples,model_choice)

    # Round the predictions to get integer values of 0 or 1
    predictions = np.round(predictions.reshape(-1,8,8)).astype(int)

    if use_corrective_algorithm == 1:
        # Pass through corrective algorithm
        predictions = helper.corrective_algorithm_for_dungeon(predictions)

    # convert list to JSON string and send as response
    return {"data" : json.dumps(predictions.tolist())}

@app.post("/generate_rooms")
async def generate_rooms(num_of_samples: int = Form(...), model_choice: int = Form(...), use_corrective_algorithm: int = Form(...)):
    #Generate appropriate predictions as per the given parameters
    generated_levels = generate.generate_rooms(num_of_samples,model_choice)

    # Round the predictions to get integer values between 0-8
    generated_levels = np.round(np.clip(generated_levels,0,8)).astype(int)

    if use_corrective_algorithm == 1:
        # Pass through corrective algorithm
        generated_levels = helper.corrective_algorithm_for_rooms(generated_levels)

    # convert list to JSON string and send as response
    return {"data" : json.dumps(generated_levels.tolist())}

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='127.0.0.1')