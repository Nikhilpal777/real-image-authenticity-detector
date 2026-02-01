import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from gradcam import make_occlusion_heatmap, overlay_heatmap

model = tf.keras.models.load_model("ai_image_detector_fixed.h5")

def predict_image(img_path):

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0][0]

    ai_percent = prediction * 100
    real_percent = (1 - prediction) * 100

    # explainability
    heatmap = make_occlusion_heatmap(img_array, model)
    heatmap_path = overlay_heatmap(img_path, heatmap)

    if prediction > 0.65:
        label = "AI Generated Image"
        confidence = ai_percent
    elif prediction < 0.35:
        label = "Real Image"
        confidence = real_percent
    else:
        label = "Uncertain"
        confidence = max(ai_percent, real_percent)

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "heatmap": heatmap_path
    }
