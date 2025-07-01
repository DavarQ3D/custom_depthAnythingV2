import coremltools as ct
import cv2
import numpy as np
import torch
from PIL import Image
from depth_anything_v2.dpt import DepthAnythingV2
import os
from depth_anything_v2.util import transform

#=============================================================================================================

def loadTorchModel(modelPath, encoder):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }    
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(modelPath, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    return depth_anything    

#=============================================================================================================

def center_crop_or_pad(img: np.ndarray, desiredRow: int, desiredCol: int) -> np.ndarray:

    h, w = img.shape[:2]

    # centre crop if the dimension is too large 
    if h > desiredRow:
        top = (h - desiredRow) // 2
        img = img[top : top + desiredRow, :, :]
        h = desiredRow
    if w > desiredCol:
        left = (w - desiredCol) // 2
        img = img[:, left : left + desiredCol, :]
        w = desiredCol

    # symmetric padding if the dimension is too small 
    pad_top    = (desiredRow - h) // 2
    pad_bottom = desiredRow - h - pad_top
    pad_left   = (desiredCol - w) // 2
    pad_right  = desiredCol - w - pad_left

    if any(p > 0 for p in (pad_top, pad_bottom, pad_left, pad_right)):
        img = cv2.copyMakeBorder(
            img,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value = 0
        )

    return img

#=============================================================================================================

def inferFromTorch(model, image, input_size):
    return model.infer_image(image, input_size, doResize=False)

#=============================================================================================================

def inferFromCoreml(mlProg, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_input = Image.fromarray(rgb)
    pred = mlProg.predict({"image": pil_input})
    depth = np.array(pred["depth"], dtype=np.float32)
    return depth

#=============================================================================================================

def fp(value, precision=4):
    return f"{value:.{precision}f}" 

def normalize(image):
    return (image - image.min()) / (image.max() - image.min() + 1e-8)

def denormalize(image):
    return (image * 255).astype(np.uint8)

def analyzeAndPrepVis(ref, pred, mode = "color"):

    assert mode in ("color", "grayscale")
    print("ref ---> min:", fp(ref.min()), ", max:", fp(ref.max()))
    print("pred --> min:", fp(pred.min()), ", max:", fp(pred.max()), '\n')

    ref = normalize(ref)
    pred = normalize(pred)
    err = np.abs(ref - pred)
    print("err ---> min:", fp(err.min()), ", max:", fp(err.max()), "--> RMSE:", fp(np.sqrt((err**2).mean()), 6))

    ref = denormalize(ref)
    pred = denormalize(pred)    
    err = denormalize(err)

    if mode == "grayscale":
        return cv2.hconcat([ref, pred, err])

    ref = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
    pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    err = cv2.cvtColor(err, cv2.COLOR_GRAY2BGR)
    err = cv2.applyColorMap(err, cv2.COLORMAP_JET)

    return cv2.hconcat([ref, pred, err])


#=============================================================================================================

def displayImage(title, image):
    cv2.imshow(title, image)
    key = cv2.waitKey(0)
    if key == 27:  
        cv2.destroyAllWindows()
        exit()

#=============================================================================================================


if __name__ == '__main__':

    #--------------------- load the torch model
    encoder = "vits"
    torch_model = loadTorchModel(f'checkpoints/depth_anything_v2_{encoder}.pth', encoder)

    #------------------ load the Core ML model
    mlProgram = ct.models.MLModel("./checkpoints/custom_vits_F16.mlpackage") 

    #------------------ resizer
    lower_dim = 518
    resizer = transform.Resize(
        width=lower_dim,                      
        height=lower_dim,                     
        resize_target=False,                  
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method="lower_bound",      
        image_interpolation_method=cv2.INTER_CUBIC,
    )

    def customResize(image):
        sample = {"image": image}
        sample = resizer(sample)               
        return sample["image"]   
    
    #------------------ configs
    fixedRow = lower_dim                                
    fixedCol = 686 
    img_path = "./data/tall_rgb/"
    outdir   = "./data/outputs"
    os.makedirs(outdir, exist_ok=True)
    numFiles = len(os.listdir(img_path)) 

    #------------------ inference loop
    #------------------------------------------------------------------

    for idx in range(numFiles):

        print('\n'"=========================================================")
        print(f'========= sample --> {idx} =========')
        print("=========================================================", '\n')

        rgbPath = img_path + f"RGB_{idx + 1:02d}.JPG"
        rawImage = cv2.imread(rgbPath)

        resized = customResize(rawImage)
        cropped = center_crop_or_pad(resized, fixedCol, fixedRow)

        depthTorch = inferFromTorch(torch_model, cropped, fixedRow)

        rotated = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        depthCoreml = inferFromCoreml(mlProgram, rotated)
        depthCoreml = cv2.rotate(depthCoreml, cv2.ROTATE_90_CLOCKWISE)
    
        visualRes = analyzeAndPrepVis(depthTorch, depthCoreml, mode="color")
        displayImage("visualRes", visualRes)
