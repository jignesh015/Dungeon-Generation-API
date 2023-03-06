import uvicorn
import json
from keras import models
import numpy as np 
from fastapi import FastAPI, Form

app = FastAPI()


# Load the model
dungeon_autoencoder = models.load_model('models\dungeon_autoencoder.h5')


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/generate_dungeons")
async def generate_dungeons(num_of_samples: int = Form(...), mode_of_generation: int = Form(...)):
    # num_of_samples = req.num_of_samples
    sample_arr = np.random.randint(low=0, high=2, size=(num_of_samples, 64))
    predictions = dungeon_autoencoder.predict(sample_arr)
    print(predictions.shape)
    # Round the predictions to get integer values of 0 or 1
    predictions = np.round(predictions.reshape(-1,8,8)).astype(int).tolist()

    # convert list to JSON string
    json_string = json.dumps(predictions)
    # return json_string
    return {"data" : json.dumps(predictions)}

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='127.0.0.1')