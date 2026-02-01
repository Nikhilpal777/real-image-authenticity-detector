import numpy as np
import cv2

# =========================
# OCCLUSION HEATMAP
# =========================
def make_occlusion_heatmap(img_array, model, patch=40):

    base_pred = model.predict(img_array)[0][0]

    heatmap = np.zeros((224, 224))

    for y in range(0, 224, patch):
        for x in range(0, 224, patch):

            # ensure last blocks are included
            y_end = min(y + patch, 224)
            x_end = min(x + patch, 224)

            occluded = img_array.copy()
            occluded[:, y:y_end, x:x_end, :] = 0

            pred = model.predict(occluded)[0][0]

            # difference from original prediction
            diff = abs(base_pred - pred)

            heatmap[y:y_end, x:x_end] = diff

    # normalize
    heatmap -= heatmap.min()
    heatmap /= heatmap.max() + 1e-8

    return heatmap


# =========================
# OVERLAY HEATMAP
# =========================
def overlay_heatmap(original_img_path, heatmap, alpha=0.4):

    img = cv2.imread(original_img_path)

    h, w = img.shape[:2]

    # resize heatmap to real image size
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)

    output_path = "static/occlusion_result.jpg"
    cv2.imwrite(output_path, superimposed)

    return output_path
