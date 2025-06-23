import coremltools as ct
import cv2
import numpy as np
from PIL import Image

model = ct.models.MLModel("./checkpoints/DepthAnythingV2SmallF16.mlpackage")

bgr = cv2.imread("data/camera/camera_0.png", cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
rgb_resized = cv2.resize(rgb, (518, 392), interpolation=cv2.INTER_AREA)

pil_input = Image.fromarray(rgb_resized)
pred = model.predict({"image": pil_input})

depth = np.array(pred["depth"], dtype=np.float32)
norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
cv2.imshow("Depth", norm.astype(np.uint8))
key = cv2.waitKey(0)
if key == 27:  
    cv2.destroyAllWindows()
    exit()
