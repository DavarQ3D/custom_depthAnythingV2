import coremltools as ct
import torch
import torchvision

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

if __name__ == '__main__':
    load_convert_save("mobilenet_v2.mlpackage")
    
