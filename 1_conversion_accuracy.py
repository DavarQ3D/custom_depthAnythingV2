import coremltools as ct
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.dinov2 import DinoVisionTransformer
import os
from depth_anything_v2.util import transform
import math

#=============================================================================================================

def interpolate_pos_encoding_bilinear(self, x, w, h):
    previous_dtype = x.dtype
    npatch = x.shape[1] - 1
    N = self.pos_embed.shape[1] - 1
    if npatch == N and w == h:
        return self.pos_embed
    pos_embed = self.pos_embed.float()
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    dim = x.shape[-1]
    w0 = w // self.patch_size
    h0 = h // self.patch_size
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    # DINOv2 with register modify the interpolate_offset from 0.1 to 0.0
    w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset
    # w0, h0 = w0 + 0.1, h0 + 0.1
    
    sqrt_N = math.sqrt(N)
    sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
        scale_factor=(sx, sy),
        # (int(w0), int(h0)), # to solve the upsampling shape issue
        mode="bilinear",
        antialias=self.interpolate_antialias
    )
    
    assert int(w0) == patch_pos_embed.shape[-2]
    assert int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

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

if __name__ == '__main__':

    #---------- bilinear interpolation for pos encoding vs original bicubic
    DinoVisionTransformer.interpolate_pos_encoding = interpolate_pos_encoding_bilinear

    #--------------------- load the torch model
    encoder = "vits"
    torch_model = loadTorchModel(f'checkpoints/depth_anything_v2_{encoder}.pth', encoder)

    #------------------ load the Core ML model
    customModel = True
    lower_dim = 518 if customModel else 392
    fixedRow = lower_dim                                # core ML program requires fixed input size
    fixedCol = 686 if customModel else 518
    mlProgram = ct.models.CompiledMLModel(f"./checkpoints/custom_vits_F16_{fixedRow}_{fixedCol}.mlmodelc") if customModel else ct.models.MLModel("./checkpoints/DepthAnythingV2SmallF16.mlpackage")

    #------------------ configs
    img_path = "./data/camera/"
    outdir   = "./data/outputs"
    numFiles = len(os.listdir(img_path))
    filenames = [os.path.join(img_path, f"camera_{i}.png") for i in range(numFiles)]    
    os.makedirs(outdir, exist_ok=True)

    #------------------ inference loop
    #------------------------------------------------------------------
    
    for k, filename in enumerate(filenames):

        print('\n'"=========================================================")
        print(f'========= sample --> {filename} =========')
        print("=========================================================", '\n')

        raw_image = cv2.imread(filename)
               
        resized = customResize(raw_image, lower_dim, resizeMode="lower_bound")               

        cropped = center_crop_or_pad(resized, fixedRow, fixedCol)  
        depth_torch = inferFromTorch(torch_model, cropped, fixedRow)
        depth_coreml = inferFromCoreml(mlProgram, cropped)
    
        visualRes = analyzeAndPrepVis(depth_torch, depth_coreml, mode="color")
        displayImage("visualRes", visualRes)
