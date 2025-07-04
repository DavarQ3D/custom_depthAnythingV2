import coremltools as ct
import cv2
import numpy as np
from PIL import Image
import os
from depth_anything_v2.util import transform

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

def fit(ref, target):

    ref = normalize(ref)
    target = normalize(target)



#=============================================================================================================

if __name__ == '__main__':

    #------------------ load the Core ML model
    customModel = True
    mlProgram = ct.models.MLModel("./checkpoints/custom_vits_F16.mlpackage") if customModel else ct.models.MLModel("./checkpoints/DepthAnythingV2SmallF16.mlpackage")
    # mlProgram = None

    #------------------ resizer
    lower_dim = 518 if customModel else 392
    resizer = transform.Resize(
        width=lower_dim,                      
        height=lower_dim,                     
        resize_target=False,                  
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method="lower_bound",      
        image_interpolation_method=cv2.INTER_CUBIC,
    )
    
    #------------------ configs
    fixedRow = lower_dim                                # core ML program requires fixed input size
    fixedCol = 686 if customModel else 518
    img_path = "./data/iphone_pro/"
    outdir   = "./data/outputs"
    os.makedirs(outdir, exist_ok=True)
    numFiles = len(os.listdir(img_path)) 

    #------------------ inference loop
    #------------------------------------------------------------------
    
    for idx in range(numFiles):

        print('\n'"=========================================================")
        print(f'========= sample --> {idx} =========')
        print("=========================================================", '\n')

        refDepthPath = img_path + f"ARKIT_{idx + 1:02d}.JPG"
        refDepth = cv2.imread(refDepthPath, cv2.IMREAD_GRAYSCALE)

        rgbPath = img_path + f"RGB_{idx + 1:02d}.JPG"
        raw_image = cv2.imread(rgbPath)

        sample = {"image": raw_image}
        sample = resizer(sample)               
        resized = sample["image"]       
        
        cropped = center_crop_or_pad(resized, fixedRow, fixedCol)  

        predDepth = inferFromCoreml(mlProgram, cropped)        
        predDepth = normalize(predDepth)

        # cv2.imshow("raw_image", raw_image)
        # cv2.imshow("resized", resized)
        # cv2.imshow("cropped", cropped)
        # cv2.imshow("predDepth", predDepth)
        # cv2.imshow("refDepth", refDepth)
        # key = cv2.waitKey(0)
        # if key == 27:   
        #     cv2.destroyAllWindows()
        #     exit()

        # fit(refDepth, predDepth)
    

        # displayImage("depth_coreml", predDepth)
    
        # visualRes = analyzeAndPrepVis(depth_torch, depth_coreml, mode="color")
        # displayImage("visualRes", visualRes)
