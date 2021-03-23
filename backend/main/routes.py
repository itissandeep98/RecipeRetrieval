from flask import request
from main import app
import pandas as pd
from flask_cors import cross_origin
from PIL import Image
from torchvision import transforms
import pickle
import torch


@app.route('/')
def index():
    input_image = Image.open('OurFood\\Fruits\\Apple\\app.jpg')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)


    model = torch.load('best_classifier.pt')
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    with torch.no_grad():
        output = model(input_batch)
    with open("food_labels.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]



    top1_prob, top1_catid = torch.topk(probabilities, 1)
    print(categories[top1_catid[0]])

    return {"response": True}
