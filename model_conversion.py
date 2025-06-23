import coremltools as ct
import torch
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

if __name__ == '__main__':

    #==================== load torch model
    encoder = "vits"
    torch_model = loadTorchModel(f'checkpoints/depth_anything_v2_{encoder}.pth', encoder)
    torch_model.eval()

    #==================== handle input and trace the model
    channels = 3
    example_input = torch.rand(1, channels, 518, 518)
    traced_model = torch.jit.trace(torch_model, example_input)
    shp = (1, channels, ct.RangeDim(lower_bound=518, upper_bound=1988), ct.RangeDim(lower_bound=518, upper_bound=1988)) 
    input_shape = ct.Shape(shape=shp)

    #==================== convert the model

    mlProg = ct.convert(traced_model,
                        convert_to="mlprogram",
                        compute_units=ct.ComputeUnit.ALL,          # CPU, GPU, Neural Engine
                        compute_precision=ct.precision.FLOAT16,    # not only supported by CPU and GPU, but also by Neural Engine
                        inputs=[ct.TensorType(name="image", shape=input_shape)],
                        outputs=[ct.TensorType(name="depth")])
    
    mlProg.save(f'checkpoints/custom_{encoder}_F16.mlpackage')