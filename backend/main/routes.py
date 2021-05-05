from main.models import *
from flask import request
from main import app
from PIL import Image
from torchvision import transforms
import torch
import urllib


@app.route('/', methods=['POST'])
def index():
    input_image = Image.open(urllib.request.urlopen(request.json['url']))
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    model = torch.load('main/Data/best_classifier.pt', map_location='cpu')
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    with torch.no_grad():
        output = model(input_batch)
    with open("main/Data/food_labels.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top1_prob, top1_catid = torch.topk(probabilities, 1)

    return {"response": categories[top1_catid[0]]}


@app.route('/results', methods=['POST'])
def comparator():
    sentence_query = request.json['query']

    # Performing preprocessing(splitting, uppercase, stripping space from endpoints) on the operand input
    result = getResults(sentence_query)
    return result
