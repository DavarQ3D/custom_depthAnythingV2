import coremltools as ct
import torch
import torch.nn as nn
from depth_anything_v2.dpt import DepthAnythingV2

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

class DepthWrapper(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model
    def forward(self, x):
        y = self.base(x)          # y.shape == [B, H, W]
        return y.unsqueeze(1)     # new shape == [B, 1, H, W]

if __name__ == '__main__':

    #==================== load torch model
    encoder = "vits"
    torch_model = loadTorchModel(f'checkpoints/depth_anything_v2_{encoder}.pth', encoder)
    torch_model.eval()
    wrapped = DepthWrapper(torch_model)

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