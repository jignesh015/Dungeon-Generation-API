import uvicorn
import json
import random
from keras import models
import numpy as np 
from scipy import ndimage


# This function takes a 3D array as input, 
# chooses given num of samples from the array randomly
# and returns those samples as a 2D array
def generate_random_sample(dataset, num_of_samples):
    # Randomly select the given number of samples from the dataset
    sample_indices = np.random.choice(dataset.shape[0], num_of_samples, replace=False)
    return dataset[sample_indices, :, :].reshape(-1, dataset.shape[1]*dataset.shape[1])

# This function takes a binary dungeon dataset of shape (n,8,8) as input
# performs a corrective algo on it
# and returns the output as an array of same shape
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

    try:
        min_dungeon_len = 15
        # If the largest connected region does not have at least 'min_dungeon_len' pixels, 
        # grow it by one pixel in all directions until it does
        grown_dataset = corrected_dataset.copy()
        while np.sum(grown_dataset) < min_dungeon_len:
            grown_dataset = np.vstack((np.zeros_like(grown_dataset[0,:]), grown_dataset[:-1,:]))
            grown_dataset = np.vstack((grown_dataset[1:,:], np.zeros_like(grown_dataset[0,:])))
            grown_dataset = np.hstack((np.zeros_like(grown_dataset[:,0]).reshape(-1,1), grown_dataset[:,:-1]))
            grown_dataset = np.hstack((grown_dataset[:,1:], np.zeros_like(grown_dataset[:,0]).reshape(-1,1)))

        # Check if the largest connected region is already touching the edge of the image
        if (grown_dataset[0,:] == 1).any() or (grown_dataset[-1,:] == 1).any() or (grown_dataset[:,0] == 1).any() or (grown_dataset[:,-1] == 1).any():
            return grown_dataset

        # If the largest connected region is not touching the edge of the image, 
        # grow it by one pixel in all directions until it does
        while (grown_dataset[0,:] == 0).all():
            grown_dataset = np.vstack((np.zeros_like(grown_dataset[0,:]), grown_dataset[:-1,:]))
        while (grown_dataset[-1,:] == 0).all():
            grown_dataset = np.vstack((grown_dataset[1:,:], np.zeros_like(grown_dataset[0,:])))
        while (grown_dataset[:,0] == 0).all():
            grown_dataset = np.hstack((np.zeros_like(grown_dataset[:,0]).reshape(-1,1), grown_dataset[:,:-1]))
        while (grown_dataset[:,-1] == 0).all():
            grown_dataset = np.hstack((grown_dataset[:,1:], np.zeros_like(grown_dataset[:,0]).reshape(-1,1)))
    except ValueError: 
        print("---------------Encountered some issue!---------------")
        return np.zeros((1, 1, 1)).astype(int)
        
    return grown_dataset

# This function takes the room dataset of shape (n,16,16)
# performs a corrective algo on it
# returns the output as an array of same shape
def corrective_algorithm_for_rooms(predicted_dataset):
  # Copy the content of provided dataset in to new array
  corrected_dataset = predicted_dataset.copy()
  # Iterate over each room grid to perform corrective algo
  for i in range(corrected_dataset.shape[0]):
    gridValues = corrected_dataset[i]
    size = len(gridValues)
    for i in range(size):
        for j in range(size):
            cellValue = gridValues[i][j]
            _sanityCondition = True
            _replacementValue = 0
            
            if cellValue == 0:
                pass
            elif cellValue == 1:
                # Remove walls which are not on the edges
                _sanityCondition = i == 0 or j == 0 or i == size - 1 or j == size - 1
            elif cellValue == 5:
                # Remove bookshelves which are not next to the walls
                _sanityCondition = i == 1 or j == 1 or i == size - 2 or j == size - 2
            elif cellValue == 6:
                # Remove torches which are not on the walls
                _sanityCondition = i == 1 or j == 1 or i == size - 2 or j == size - 2
            elif cellValue == 7:
                # Remove doors which are not on the edges
                _sanityCondition = i == 0 or j == 0 or i == size - 1 or j == size - 1
            
            # Replace everything on the edges which is not wall or door with a wall
            if i == 0 or j == 0 or i == size - 1 or j == size - 1:
                _sanityCondition = (cellValue == 1 or cellValue == 7)
                _replacementValue = 1
            
            gridValues[i][j] = cellValue if _sanityCondition else _replacementValue
    
    #Make sure there are at least two doors in the room
    min_door_count = 2
    edge_indices = [(0, j) for j in range(size)] + [(i, 0) for i in range(1, size-1)] + \
      [(size-1, j) for j in range(1, size-1)] + [(i, size-1) for i in range(1, size-1)]
    door_count = 0
    for i, j in edge_indices:
        if gridValues[i][j] == 7:
            door_count += 1
    if door_count < min_door_count:
        missing_door_count = min_door_count - door_count
        random_edge_indices = np.random.choice(len(edge_indices), size=missing_door_count, replace=False)
        for idx in random_edge_indices:
          i, j = edge_indices[idx]
          gridValues[i][j] = 7
    
    # Put a hard limit on other objects
    objectsMaxAllowed = {2:6, 3:12, 4:4, 5:6, 6:6, 8:10}
    
    for keyVal in objectsMaxAllowed.items():
        targetValue = keyVal[0]
        maxAllowed = keyVal[1]
        
        # Find the positions of all occurrences of the target value
        positions = [(x, y) for x in range(size) for y in range(size) if gridValues[x][y] == targetValue]
        
        # If there are more than maxAllowed occurrences, randomly choose which ones to keep
        if len(positions) > maxAllowed:
            positionsToConvert = random.sample(positions, len(positions) - maxAllowed)
            
            # Convert the values in the selected positions to 0
            for pos in positionsToConvert:
                gridValues[pos[0]][pos[1]] = 0   
  return corrected_dataset
