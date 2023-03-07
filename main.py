import uvicorn
import json
import random
from keras import models
import numpy as np 
from fastapi import FastAPI, Form

app = FastAPI()

#Load the datasets
dungeon_data = np.load('datasets\dungeons_dataset.npy')
room_data = np.load('datasets\\rooms_dataset.npy')

# Replace the invalid values with 1
dungeon_data[np.logical_and(dungeon_data != 0, dungeon_data != 1)] = 1

# Load the dungeon models
dungeon_autoencoder = models.load_model('models\dungeon_autoencoder.h5')

#Load the room models
rooms_cnn =  models.load_model('models\\rooms_cnn_model.h5')

def generate_random_sample(dataset, num_of_samples):
    #Randomly select the given number of samples from the dataset
    sample_indices = np.random.choice(dataset.shape[0], num_of_samples, replace=False)
    return dataset[sample_indices, :, :].reshape(-1, dataset.shape[1]*dataset.shape[1])


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/generate_dungeons")
async def generate_dungeons(num_of_samples: int = Form(...), mode_of_generation: int = Form(...)):
    # Generate a sample dataset to pass to the model
    # 0 = Generate sample with random values
    # 1 = Select random levels from the dungeon dataset 
    sample_arr = np.random.randint(low=0, high=2, size=(num_of_samples, 64)) if mode_of_generation == 0 \
        else  generate_random_sample(dungeon_data,num_of_samples)
    
    #Pass sample to the model
    predictions = dungeon_autoencoder.predict(sample_arr)

    # Round the predictions to get integer values of 0 or 1
    predictions = np.round(predictions.reshape(-1,8,8)).astype(int).tolist()

    # convert list to JSON string and send as response
    return {"data" : json.dumps(predictions)}

@app.post("/generate_rooms")
async def generate_rooms(num_of_samples: int = Form(...), mode_of_generation: int = Form(...)):
    # Initialize an array to store the output from the model
    generated_levels = np.zeros((num_of_samples, 16, 16))

    for i in range(num_of_samples):
        # Generate a sample dataset to pass to the model
        random_int = random.randint(0, len(room_data) - 1)
        sample_data = np.random.randint(low=0, high=9, size=(1, 16,16,1)) / 8.0 if mode_of_generation == 0 \
            else  room_data[random_int].reshape(1,16,16,1) / 8.0

        #Pass the sample data to the model
        level = rooms_cnn.predict(sample_data,verbose=0)
        level = np.squeeze(level)
        level = np.clip(level, 0, 8)  # clip to 0-8
        generated_levels[i] = level

    # Scale the generated levels to 0-8
    max_value = np.max(generated_levels)
    if max_value != 0:
        generated_levels *= 8.0/max_value
    generated_levels = np.round(np.clip(generated_levels,0,8)).astype(int).tolist()

    # convert list to JSON string and send as response
    return {"data" : json.dumps(generated_levels)}

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='127.0.0.1')