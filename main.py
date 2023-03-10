import uvicorn
import json
import random
from keras import models
import numpy as np 
from scipy import ndimage
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

def corrective_algorithm_for_dungeon(predicted_dataset):
  # Label the connected regions of 1's
  labels, num_labels = ndimage.label(predicted_dataset)
  # Count the number of pixels in each connected region
  counts = np.bincount(labels.ravel())
  # Find the label of the largest connected region
  largest_label = np.argmax(counts[1:]) + 1

  # Create a new array where all pixels outside the largest connected region are set to 0,
  # and all pixels inside the largest connected region are set to 1

  corrected_dataset = np.zeros_like(predicted_dataset)
  corrected_dataset[labels == largest_label] = 1

  # If the largest connected region does not have at least 10 pixels, grow it by one pixel in all directions until it does
  grown_dataset = corrected_dataset.copy()
  while np.sum(grown_dataset) < 10:
    grown_dataset = np.vstack((np.zeros_like(grown_dataset[0,:]), grown_dataset[:-1,:]))
    grown_dataset = np.vstack((grown_dataset[1:,:], np.zeros_like(grown_dataset[0,:])))
    grown_dataset = np.hstack((np.zeros_like(grown_dataset[:,0]).reshape(-1,1), grown_dataset[:,:-1]))
    grown_dataset = np.hstack((grown_dataset[:,1:], np.zeros_like(grown_dataset[:,0]).reshape(-1,1)))

  # Check if the largest connected region is already touching the edge of the image
  if (grown_dataset[0,:] == 1).any() or (grown_dataset[-1,:] == 1).any() or (grown_dataset[:,0] == 1).any() or (grown_dataset[:,-1] == 1).any():
    return grown_dataset

  # If the largest connected region is not touching the edge of the image, grow it by one pixel in all directions until it does
  grown_dataset = grown_dataset.copy()
  while (grown_dataset[0,:] == 0).all():
    grown_dataset = np.vstack((np.zeros_like(grown_dataset[0,:]), grown_dataset[:-1,:]))
  while (grown_dataset[-1,:] == 0).all():
    grown_dataset = np.vstack((grown_dataset[1:,:], np.zeros_like(grown_dataset[0,:])))
  while (grown_dataset[:,0] == 0).all():
    grown_dataset = np.hstack((np.zeros_like(grown_dataset[:,0]).reshape(-1,1), grown_dataset[:,:-1]))
  while (grown_dataset[:,-1] == 0).all():
    grown_dataset = np.hstack((grown_dataset[:,1:], np.zeros_like(grown_dataset[:,0]).reshape(-1,1)))

  return grown_dataset


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/generate_dungeons")
async def generate_dungeons(num_of_samples: int = Form(...), use_corrective_algorithm: int = Form(...)):
    # Generate a sample dataset to pass to the model
    sample_arr = np.random.randint(low=0, high=2, size=(num_of_samples, 64))
    
    #Pass sample to the model
    predictions = dungeon_autoencoder.predict(sample_arr)

    # Round the predictions to get integer values of 0 or 1
    predictions = np.round(predictions.reshape(-1,8,8)).astype(int)

    if use_corrective_algorithm == 1:
        # Pass through corrective algorithm
        predictions = corrective_algorithm_for_dungeon(predictions)

    # convert list to JSON string and send as response
    return {"data" : json.dumps(predictions.tolist())}

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