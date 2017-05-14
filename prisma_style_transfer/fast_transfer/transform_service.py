import requests
import os
import numpy as np
import scipy.misc
from flask import Flask, abort, request, send_file
from PIL import Image
from cStringIO import StringIO
from glob import glob
from io import BytesIO
from .gen_artwork_api import TransferNet

app = Flask(__name__)
transformer_dict = dict()
for model_path in glob("models/*.model"):
    model_name = os.path.basename(model_path).split(".")[0]
    transformer_dict[model_name] = TransferNet(model_path)

@app.route("/transform/<model_name>", methods=["POST"])
def transform(model_name):
    image_url = request.get_json()
    response = requests.get(image_url)
    if response.status_code == requests.codes.ok:
        image_data = StringIO(response.content)
        image = Image.open(image_data)
        image_np = np.array(image)
        image_transformed = transformer_dict[model_name].gen_single(image_np)
        buffer = BytesIO()
        scipy.misc.imsave(buffer, image_transformed, "png")
        buffer.seek(0)
        return send_file(buffer, mimetype="image/png")
    else:
        abort(404)
