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
            borderType=cv2.BORDER_REFLECT_101,
            # borderType=cv2.BORDER_CONSTANT,
            # value = 0
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
    err = normalize(err)

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

def customResize(image, lower_dim, resizeMode = "lower_bound"):

    assert resizeMode in ("lower_bound", "upper_bound")

    resizer = transform.Resize(
        width=lower_dim,                      
        height=lower_dim,                     
        resize_target=False,                  
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method=resizeMode,      
        image_interpolation_method=cv2.INTER_CUBIC,
    )

    sample = {"image": image}
    sample = resizer(sample)               
    return sample["image"]   

#=============================================================================================================
from pathlib import Path

def loadMatrixFromFile(path):

    path = Path(path)
    # np.loadtxt handles stripping whitespace; `dtype=np.float64` enforces double precision
    matrix = np.loadtxt(path, delimiter=',', dtype=np.float64)

    # If you ever need the transposed orientation, just call matrix.T
    return matrix

#=============================================================================================================

def ensure_multiple_of(x, multiple_of=14):
    return (np.floor(x / multiple_of) * multiple_of).astype(int)

#=============================================================================================================

def fitContentTo(pred: np.ndarray,
                 gt:   np.ndarray,
                 mask: np.ndarray | None = None,
                 *,
                 allow_shift: bool = True,
                 eps: float = 1e-8):

    if mask is None:
        mask = (gt > 0) & np.isfinite(gt) & np.isfinite(pred)

    if not mask.any():
        raise ValueError("No valid pixels in mask for scale/shift fitting")

    if allow_shift:
        # Least-squares: [pred, 1] ⋅ [s, t]ᵀ ≈ gt
        A = np.vstack([pred[mask].ravel(),
                       np.ones(mask.sum())]).T
        scale, shift = np.linalg.lstsq(A, gt[mask].ravel(),
                                       rcond=None)[0]
    else:
        # Median-based scale only (Monodepth-style)
        scale = (np.median(gt[mask]) /
                 (np.median(pred[mask]) + eps))
        shift = 0.0

    aligned = scale * pred + shift
    return aligned, scale, shift

#=============================================================================================================

def checkIfSynced(rgb, depth):
    rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    gray = normalize(gray)
    depth = normalize(depth)
    diff = np.abs(gray - depth)
    gray = cv2.resize(gray, (gray.shape[1] * 4, gray.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
    depth = cv2.resize(depth, (depth.shape[1] * 4, depth.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
    diff = cv2.resize(diff, (diff.shape[1] * 4, diff.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow("gray", gray)
    # cv2.imshow("depth", depth)
    cv2.imshow("diff", diff)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
        exit()

#=============================================================================================================

if __name__ == '__main__':

    #--------------------- load the torch model
    encoder = "vits"
    torch_model = loadTorchModel(f'checkpoints/depth_anything_v2_{encoder}.pth', encoder)

    #------------------ configs
    img_path = "./data/iphone_images/"
    lidar_path = "./data/iphone_pro_lidar/"
    outdir   = "./data/outputs"
    os.makedirs(outdir, exist_ok=True)
    numFiles = len(os.listdir(img_path)) 

    #------------------ inference loop
    #------------------------------------------------------------------
    
    for idx in range(numFiles):

        print('\n'"=========================================================")
        print(f'========= sample --> {idx} =========')
        print("=========================================================", '\n')

        # rgbPath = img_path + f"RGB_{idx+1:04d}.JPG"
        rgbPath = img_path + f"RGB_{10:04d}.JPG"
        raw_image = cv2.imread(rgbPath)

        # refDepthpath = lidar_path + f"DepthValues_{idx+1:04d}.txt"
        refDepthpath = lidar_path + f"DepthValues_{10:04d}.txt"
        gt = loadMatrixFromFile(refDepthpath)
        gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)

        # checkIfSynced(raw_image, gt)

        resized = cv2.resize(raw_image, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)

        r = ensure_multiple_of(resized.shape[0], multiple_of=14)
        c = ensure_multiple_of(resized.shape[1], multiple_of=14)
        cropped = resized[0 : r, 0 : c, :]
        gt = gt[0 : r, 0 : c]

        pred = inferFromTorch(torch_model, cropped, c)
        eps = 1e-8
        pred = 1.0 / (pred + eps)  

        pred, scale, shift = fitContentTo(pred, gt)

        print(f"Scale: {fp(scale)}, Shift: {fp(shift)}")

        diff = np.abs(gt - pred)
        cv2.imshow("cropped", cropped)
        cv2.imshow("gt", gt)
        cv2.imshow("pred", pred)
        cv2.imshow("diff", diff)
        key = cv2.waitKey(0)
        if key == 27:  
            cv2.destroyAllWindows()
            exit()


        





        





