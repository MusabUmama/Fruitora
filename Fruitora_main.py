
from flask import Flask, request, render_template
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import random
from Test_model import read_file_as_image
import base64

app = Flask(__name__, static_url_path='/static')

fruit_detection_model = tf.keras.models.load_model("D:\DSGP Models\models\Fruits detection model")#256,256
apple_quality_model = tf.keras.models.load_model("D:\DSGP Models\models\Apple_model2_256")#256,256
banana_quality_model = tf.keras.models.load_model("D:\DSGP Models\models\Banana_model2_256")#256,256
guava_quality_model = tf.keras.models.load_model("D:\DSGP Models\models\Guava_model2-256")#256,256
Lime_quality_model= tf.keras.models.load_model("D:\DSGP Models\models\Lime model")
Orange_quality_model=tf.keras.models.load_model("D:\DSGP Models\models\Orange model")
Pomegranate_quality_model=tf.keras.models.load_model("D:\DSGP Models\models\Pomegrante model")

fruit_quality_models = {
    "Apple": apple_quality_model,
    "Banana": banana_quality_model,
    "Guava": guava_quality_model,
    "Lime":Lime_quality_model,
    "Orange":Orange_quality_model,
    "Pomegranate":Pomegranate_quality_model
    #"Unknown":"This is not a Fruit"
    # Add more fruit types and their corresponding quality models as needed
}

fruit_class_names = {
    "Apple": [ "Bad quality","Mix_quality","Good quality"],
    "Banana": ["Bad quality","Good quality" ],
    "Guava": ["Bad quality","Good quality" ],
    "Lime": ["Bad quality","Good quality" ],
    "Orange": ["Bad quality","Good quality" ],
"Pomegranate":["Good quality","Bad quality",]
    # Add more fruit types and their corresponding class names as needed
}





def combine_models(fruit_detection_model, fruit_quality_models, fruit_image, threshold=0.80):
    # Use the fruit detection model to identify the type of fruit
    fruit_type_probs = fruit_detection_model.predict(fruit_image)
    fruit_type_index = np.argmax(fruit_type_probs)
    print(fruit_type_probs)
    print(fruit_type_index)
    print(np.max(fruit_type_probs))
    fruit_type = ""
    if fruit_type_index == 0:
        fruit_type = "Apple"
    elif fruit_type_index == 1:
       fruit_type = "Banana"
    elif fruit_type_index == 2:
        fruit_type = "Guava"
    elif fruit_type_index == 3:
        fruit_type = "Lime"
    elif fruit_type_index == 4:
        fruit_type = "Orange"
    elif fruit_type_index == 5:
        fruit_type = "Pomegranate"
    else:
        fruit_type = "Unknown"

    # Check if the predicted fruit class probability is below the threshold
    if np.max(fruit_type_probs) < threshold:
        return "Unknown", None

    if fruit_type == "Unknown":
        return fruit_type, None

    # Select the appropriate fruit quality detection model for this fruit type
    fruit_quality_model = fruit_quality_models[fruit_type]

    # Use the selected fruit quality detection model to determine the quality of the fruit
    fruit_quality = fruit_quality_model.predict(fruit_image)

    return fruit_type, fruit_quality

import cv2

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))

    return image


@app.route('/home')
def home():
    return render_template("Detection page.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    img = np.array(Image.open(file))
    img = cv2.resize(img, (256, 256))
    img_batch = np.expand_dims(img, 0)
    fruit_type, fruit_quality = combine_models(fruit_detection_model, fruit_quality_models, img_batch)

    if fruit_type == "Unknown":
        return render_template("Detection page.html", Error_msg='Please input a valid fruit image (Apple,Banana,Guava,Lime, Orange, Pomegranate)')

    if fruit_type in fruit_class_names.keys():
        class_names = fruit_class_names[fruit_type]
        predicted_class = class_names[np.argmax(fruit_quality[0])]
        confidence = np.max(fruit_quality[0])
        good_quality_percentage = confidence * 100
        bad_quality_percentage = 100 - good_quality_percentage

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply a threshold to the image to separate the black spots
        ret, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)

        # Find the contours of the black spots in the image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if predicted_class == "Good quality":
            # Draw the contours on the original image

            return render_template("Detection page.html", Quality_class='This {} is in {}'.format(fruit_type, predicted_class),
                                   Quality_percentage="{}".format(int(good_quality_percentage)),
                                   Msg="Yes, you can eat this",img_src="data:image/png;base64,{}".format(base64.b64encode(cv2.imencode('.png', img)[1]).decode()))

        elif predicted_class == "Bad quality":
            if 10 <= bad_quality_percentage <= 40:
                # Draw the contours on the original image
                cv2.drawContours(img, contours, -1, (0,0, 255), 2)
                img = cv2.resize(img, (256, 256))

                return render_template("Detection page.html", Quality_class='This {} is in {}'.format(fruit_type, predicted_class),
                                       Quality_percentage="{}".format(int(bad_quality_percentage)),
                                       Msg="Don't eat this, it's rotten!!!",
                                       img_src="data:image/png;base64,{}".format(base64.b64encode(cv2.imencode('.png', img)[1]).decode()))
            else:
                # Draw the contours on the original image
                cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
                img = cv2.resize(img, (256, 256))

                return render_template("Detection page.html", Quality_class='This {} is in {}'.format(fruit_type, predicted_class),
                                       Quality_percentage="{}".format(int(bad_quality_percentage)),
                                       Msg="Think before eating!!!",
                                       img_src="data:image/png;base64,{}".format(base64.b64encode(cv2.imencode('.png', img)[1]).decode()))


if __name__ == "__main__":
    app.run(host='localhost', port=8000)



