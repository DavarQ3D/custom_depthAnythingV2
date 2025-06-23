import coremltools as ct
import cv2
import numpy as np
from PIL import Image
from google.protobuf.json_format import MessageToJson

model = ct.models.MLModel("./checkpoints/custom_vits_F16.mlpackage")

writeSpecOnDisk = False

if writeSpecOnDisk:
    spec = model.get_spec()
    json_str = MessageToJson(spec)
    with open("model_spec.json", "w") as f:
        f.write(json_str)
    exit()

bgr = cv2.imread("data/camera/camera_0.png", cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
rgb_resized = cv2.resize(rgb, (518, 518), interpolation=cv2.INTER_AREA)

pil_input = Image.fromarray(rgb_resized)
pred = model.predict({"image": pil_input})

depth = np.array(pred["depth"], dtype=np.float32)
norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
cv2.imshow("Depth", norm.astype(np.uint8))
key = cv2.waitKey(0)
if key == 27:  
    cv2.destroyAllWindows()
    exit()
