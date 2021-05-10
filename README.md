<h1 align="center"> IR Project</h1>
<h2 align="center">Team 34</h2>

# Introduction
The world is a collection of multiple cultures, and with them come multiple languages and dialects. This can be rather challenging for current technologies to deal with, if they rely on text-based inputs. With advancements in machine learning and object recognition, it is the need of the hour to switch to image-based input systems. We propose a recipe retrieval system that uses images of ingredients as an input. The user is simply required to use a camera to click pictures of available ingredients and upload it to the system. The designed Information Retrieval system will produce a list of relevant recipes based on a ranking system. The system has been designed for efficient use in the kitchen as well as in the grocery store.

# Instructions for running both Apps

## Frontend <https://itissandeep98.github.io/RecipeRetrieval>

``` 
- Just visit the above URL and start backend locally or on google colab and obtain the API URL.
```
OR
```
- cd frontend
- npm install
- npm run start
```

## Backend

```
- cd backend
- virtualenv venv    ## Only first time
- source venv/bin/activate
- pip install -r requirements.txt
- python3 run.py  #Paste the url obtained in the browser window
```
OR
```
- Open IR_backend.ipynb preferably on google colab or something similar
- Run all cells and install any package if required
- You will obtain a ngrok's API url, paste that URL in the browser
```

### Extra Data
You will require to have the following extra files in order to run the backend sucessfully

<https://drive.google.com/drive/folders/1HsIX7RvVWkylzgJLQhMl-7V3oXWvxaTq?usp=sharing>
