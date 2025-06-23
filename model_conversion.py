import coremltools as ct
import torch
from depth_anything_v2.dpt import DepthAnythingV2

def load_convert_save(outPath):

    torch_model = torchvision.models.mobilenet_v2()
    torch_model.eval() 

    example_input = torch.rand(1, 3, 256, 256)
    traced_model = torch.jit.trace(torch_model, example_input)

    # Convert using the same API. Note that we need to provide "inputs" for pytorch conversion.
    convertedModel = ct.convert(traced_model,
                        convert_to="mlprogram",
                        compute_units=ct.ComputeUnit.ALL,
                        compute_precision=ct.precision.FLOAT16,
                        inputs=[ct.TensorType(name="input", shape=example_input.shape)])

    convertedModel.save(outPath)


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
    step = 14
    hm = 37   # min 518
    wm = 50
    height = hm * step
    width = wm * step
    chansels = 3
    example_input = torch.rand(1, chansels, height, width)
    traced_model = torch.jit.trace(torch_model, example_input)
    
    #==================== convert the model