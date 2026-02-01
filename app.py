from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from gradcam import make_occlusion_heatmap, overlay_heatmap

app = Flask(__name__)

# load model
model = tf.keras.models.load_model("ai_image_detector_fixed.h5")

# folders
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    confidence = None
    uploaded_image = None
    heatmap_image = None

    if request.method == "POST":

        file = request.files["image"]

        # save inside static so browser can load it
        save_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(save_path)

        # URL path for browser
        uploaded_image = "/" + save_path

        # preprocess image
        img = image.load_img(save_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # prediction
        prediction = model.predict(img_array)[0][0]

        ai_percent = prediction * 100
        real_percent = (1 - prediction) * 100

        if prediction > 0.65:
            result = "AI Generated Image"
            confidence = round(ai_percent, 2)
        elif prediction < 0.35:
            result = "Real Image"
            confidence = round(real_percent, 2)
        else:
            result = "Uncertain"
            confidence = round(max(ai_percent, real_percent), 2)

        # heatmap
        heatmap = make_occlusion_heatmap(img_array, model)
        heatmap_path = overlay_heatmap(save_path, heatmap)

        heatmap_image = "/" + heatmap_path

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        uploaded_image=uploaded_image,
        heatmap_image=heatmap_image
    )

if __name__ == "__main__":
    app.run(debug=True)
