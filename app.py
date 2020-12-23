from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from flask import jsonify, make_response
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = "./model_resnet.h5"

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()          # Necessary
# print("Model loaded. Start serving...")

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights="imagenet", include_top=True)
#model.save("")
#print("Model loaded. Check http://127.0.0.1:5000/")

names = ["AFRICAN FIREFINCH","ALBATROSS","ALEXANDRINE PARAKEET","AMERICAN AVOCET","AMERICAN BITTERN","AMERICAN COOT","AMERICAN GOLDFINCH","AMERICAN KESTREL","AMERICAN PIPIT","AMERICAN REDSTART","ANHINGA","ANNAS HUMMINGBIRD","ANTBIRD","ARARIPE MANAKIN","BALD EAGLE","BALTIMORE ORIOLE","BANANAQUIT","BAR-TAILED GODWIT","BARN OWL","BARN SWALLOW","BAY-BREASTED WARBLER","BEARDED BARBET","BELTED KINGFISHER","BIRD OF PARADISE","BLACK FRANCOLIN","BLACK SKIMMER","BLACK SWAN","BLACK THROATED WARBLER","BLACK VULTURE","BLACK-CAPPED CHICKADEE","BLACK-NECKED GREBE","BLACK-THROATED SPARROW","BLACKBURNIAM WARBLER","BLUE GROUSE","BLUE HERON","BOBOLINK","BROWN NOODY","BROWN THRASHER","CACTUS WREN","CALIFORNIA CONDOR","CALIFORNIA GULL","CALIFORNIA QUAIL","CANARY","CAPE MAY WARBLER","CAPUCHINBIRD","CARMINE BEE-EATER","CASPIAN TERN","CASSOWARY","CHARA DE COLLAR","CHIPPING SPARROW","CINNAMON TEAL","COCK OF THE  ROCK","COCKATOO","COMMON GRACKLE","COMMON HOUSE MARTIN","COMMON LOON","COMMON POORWILL","COMMON STARLING","COUCHS KINGBIRD","CRESTED AUKLET","CRESTED CARACARA","CROW","CROWNED PIGEON","CUBAN TODY","CURL CRESTED ARACURI","D-ARNAUDS BARBET","DARK EYED JUNCO","DOWNY WOODPECKER","EASTERN BLUEBIRD","EASTERN MEADOWLARK","EASTERN ROSELLA","EASTERN TOWEE","ELEGANT TROGON","ELLIOTS  PHEASANT","EMPEROR PENGUIN","EMU","EURASIAN MAGPIE","EVENING GROSBEAK","FLAME TANAGER","FLAMINGO","FRIGATE","GILA WOODPECKER","GILDED FLICKER","GLOSSY IBIS","GOLD WING WARBLER","GOLDEN CHEEKED WARBLER","GOLDEN CHLOROPHONIA","GOLDEN EAGLE","GOLDEN PHEASANT","GOULDIAN FINCH","GRAY CATBIRD","GRAY PARTRIDGE","GREEN JAY","GREY PLOVER","GUINEAFOWL","HAWAIIAN GOOSE","HOODED MERGANSER","HOOPOES","HORNBILL","HOUSE FINCH","HOUSE SPARROW","LEARS MACAW","IMPERIAL SHAQ","INCA TERN","INDIGO BUNTING","JABIRU","JAVAN MAGPIE","KILLDEAR","KING VULTURE","LARK BUNTING","LILAC ROLLER","LONG-EARED OWL","MALACHITE KINGFISHER","MALEO","MALLARD DUCK","MANDRIN DUCK","MARABOU STORK","MASKED BOOBY","MIKADO  PHEASANT","MOURNING DOVE","MYNA","NICOBAR PIGEON","NORTHERN CARDINAL","NORTHERN FLICKER","NORTHERN GANNET","NORTHERN GOSHAWK","NORTHERN JACANA","NORTHERN MOCKINGBIRD","NORTHERN PARULA","NORTHERN RED BISHOP","OCELLATED TURKEY","OSPREY","OSTRICH","PAINTED BUNTIG","PALILA","PARADISE TANAGER","PARUS MAJOR","PEACOCK","PELICAN","PEREGRINE FALCON","PHILIPPINE EAGLE","PINK ROBIN","PUFFIN","PURPLE FINCH","PURPLE GALLINULE","161:HARPY EAGLE","PURPLE MARTIN","PURPLE SWAMPHEN","QUETZAL","RAINBOW LORIKEET","RAZORBILL","RED FACED CORMORANT","RED FACED WARBLER","RED HEADED DUCK","RED HEADED WOODPECKER","RED HONEY CREEPER","RED THROATED BEE EATER","RED WINGED BLACKBIRD","RED WISKERED BULBUL","RING-NECKED PHEASANT","ROADRUNNER","ROBIN","ROCK DOVE","ROSY FACED LOVEBIRD","GYRAFALCON","ROUGH LEG BUZZARD","RUBY THROATED HUMMINGBIRD","RUFOUS KINGFISHER","RUFUOS MOTMOT","SAND MARTIN","SCARLET IBIS","SCARLET MACAW","SHOEBILL","SMITHS LONGSPUR","SNOWY EGRET","SNOWY OWL","SORA","SPANGLED COTINGA","SPLENDID WREN","SPOON BILED SANDPIPER","SPOONBILL","STEAMER DUCK","STORK BILLED KINGFISHER","STRAWBERRY FINCH","STRIPPED SWALLOW","SUPERB STARLING","TAIWAN MAGPIE","TAKAHE","TASMANIAN HEN","TEAL DUCK","TIT MOUSE","TOUCHAN","TOWNSENDS WARBLER","TREE SWALLOW","TRUMPTER SWAN","TURKEY VULTURE","TURQUOISE MOTMOT","VARIED THRUSH","VENEZUELIAN TROUPIAL","VERMILION FLYCATHER","VIOLET GREEN SWALLOW","WATTLED CURASSOW","WHIMBREL","WHITE CHEEKED TURACO","WHITE NECKED RAVEN","WHITE TAILED TROPIC","WILD TURKEY","WILSONS BIRD OF PARADISE","WOOD DUCK","YELLOW CACIQUE","YELLOW HEADED BLACKBIRD"]

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won"t make correct prediction!
    x = preprocess_input(x, mode="caffe")

    preds = model.predict(x)
    label = np.argmax(preds)
    print("Label", label)
    labelName = names[label]
    print("Label Name", labelName)
    return labelName


@app.route("/", methods=["GET"])
def index():
    # Main page
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        # Get the file from post request
        f = request.files["file"]

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, "uploads", secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        resp = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = (preds.data)               # Convert to string
        return resp


if __name__ == "__main__":
    app.run(debug=True)

