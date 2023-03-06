import uvicorn
import json
from keras import models
import numpy as np 
from fastapi import FastAPI, Form

app = FastAPI()

#Load the datasets
dungeon_data = np.load('datasets\dungeons_dataset.npy')
room_data = np.load('datasets\\rooms_dataset.npy')

# Replace the invalid values with 1
dungeon_data[np.logical_and(dungeon_data != 0, dungeon_data != 1)] = 1

# Load the model
dungeon_autoencoder = models.load_model('models\dungeon_autoencoder.h5')

def generate_random_sample(dataset, num_of_samples):
    #Randomly select the given number of samples from the dataset
    sample_indices = np.random.choice(dataset.shape[0], num_of_samples, replace=False)
    return dataset[sample_indices, :, :].reshape(-1, dataset.shape[1]*dataset.shape[1])


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/generate_dungeons")
async def generate_dungeons(num_of_samples: int = Form(...), mode_of_generation: int = Form(...)):
    #Generate a random sample to pass to the model
    sample_arr = np.random.randint(low=0, high=2, size=(num_of_samples, 64)) if mode_of_generation == 0 \
        else  generate_random_sample(dungeon_data,num_of_samples)
    
    #Pass sample to the model
    predictions = dungeon_autoencoder.predict(sample_arr)

    # Round the predictions to get integer values of 0 or 1
    predictions = np.round(predictions.reshape(-1,8,8)).astype(int).tolist()

    # convert list to JSON string and send as response
    return {"data" : json.dumps(predictions)}

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='127.0.0.1')