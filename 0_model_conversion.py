import math
import coremltools as ct
import torch
import torch.nn as nn
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.dinov2 import DinoVisionTransformer

#--------------------------------------------------------------------------------------------------------------

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

DinoVisionTransformer.interpolate_pos_encoding = interpolate_pos_encoding_bilinear

#--------------------------------------------------------------------------------------------------------------

class DepthWrapper(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = x / 255.0
        x = (x - self.mean) / self.std
        y = self.base(x)           # y: (B, H, W)
        return y.unsqueeze(1)      # -> (B, 1, H, W)

#--------------------------------------------------------------------------------------------------------------

def loadTorchModel(modelPath, encoder):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }    
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(modelPath, map_location='cpu'))
    return depth_anything    

#--------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    #==================== load torch model
    encoder = "vits"
    torch_model = loadTorchModel(f'checkpoints/depth_anything_v2_{encoder}.pth', encoder)
    torch_model.eval()
    wrapped = DepthWrapper(torch_model)
    wrapped.eval()

    #==================== conversion
    example_input = torch.rand(1, 3, 518, 518)
    traced_model = torch.jit.trace(wrapped, example_input)

    mlProg = ct.convert(traced_model,
                        convert_to="mlprogram",
                        compute_units=ct.ComputeUnit.ALL,           # CPU, GPU, Neural Engine
                        compute_precision=ct.precision.FLOAT16,     # not only supported by CPU and GPU, but also by Neural Engine
                        minimum_deployment_target=ct.target.iOS16,  # required for GRAYSCALE_FLOAT16
                        inputs=[ct.ImageType(name="image", shape=example_input.shape, color_layout=ct.colorlayout.RGB)],
                        outputs=[ct.ImageType(name="depth", color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)])
    
    mlProg.save(f'checkpoints/custom_{encoder}_F16.mlpackage')