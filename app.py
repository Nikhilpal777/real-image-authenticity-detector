from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# heatmap functions
from gradcam import make_occlusion_heatmap, overlay_heatmap

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = None  # lazy load


def get_model():
    global model
    if model is None:
        print("Loading model...")
        model = tf.keras.models.load_model("ai_image_detector_fixed.h5")
        print("Model loaded")
    return model


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    uploaded_image = None
    heatmap_image = None

    if request.method == "POST":
        file = request.files["image"]
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        img = image.load_img(path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        model = get_model()
        prediction = model.predict(img_array)[0][0]

        ai_percent = prediction * 100
        real_percent = (1 - prediction) * 100

        if prediction > 0.65:
            result = "AI Generated"
            confidence = f"{ai_percent:.2f}"
        elif prediction < 0.35:
            result = "Real Image"
            confidence = f"{real_percent:.2f}"
        else:
            result = "Uncertain"
            confidence = f"{max(ai_percent, real_percent):.2f}"

        uploaded_image = path

        
        heatmap = make_occlusion_heatmap(img_array, model)
        heatmap_image = overlay_heatmap(path, heatmap)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        uploaded_image=uploaded_image,
        heatmap_image=heatmap_image
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
