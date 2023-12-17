from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import pickle

# Load the pre-trained model
model = pickle.load(open("C:\\documents\\DevOpsdemo\\final\\model\\scovid_model.pkl", "rb"))

# Function to preprocess the uploaded image
def preprocess_image(image_data):
    # Decode image data
    image = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)

    # Resize and normalize (matches the expected input shape of the model)
    image = cv2.resize(image, (224, 224)) / 255.0

    # Expand the dimensions to match the expected batch size of 1
    image = np.expand_dims(image, axis=0)

    # Return the preprocessed image
    return image

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get uploaded image file
        image_file = request.files["image"]

        # Validate file extension using tuple conversion
        allowed_extensions = ("jpg", "jpeg", "png")
        if not image_file.filename.endswith(allowed_extensions):
            return render_template("error.html", error_message="Invalid image format. Please upload a JPG, JPEG, or PNG file.")

        # Read image data
        image_data = image_file.read()

        # Preprocess the image
        processed_image = preprocess_image(image_data)

        # Predict with the model
        prediction = model.predict(processed_image)[0]
        predicted_class = np.argmax(prediction)

        # Generate result message
        if predicted_class == 1:
            result = "Predicted: COVID positive"
        else:
            result = "Predicted: COVID negative"

        # Render result page
        return render_template("result.html", result=result)
    else:
        return render_template("index.html")

if __name__ == '__main__':

    app.run(host="0.0.0.0", port=8000) #for deployment run
    #app.run(host="127.0.0.1", port=8000,debug=True) # for local run