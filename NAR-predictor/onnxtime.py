import onnxruntime as ort
import numpy as np
import time


def test_onnx_model_runtime(model_path, input_data):
    """
    Test the runtime of an ONNX model.

    Parameters:
    model_path (str): The path to the ONNX model file.
    input_data (dict): A dictionary containing input data for the model.

    Returns:
    float: The runtime of the model in seconds.
    """
    # Load the ONNX model
    session = ort.InferenceSession(model_path)

    # Get input name for the model
    input_name = session.get_inputs()[0].name

    # Measure the time taken to run the model
    start_time = time.time()
    session.run(None, {input_name: input_data})
    end_time = time.time()

    # Calculate the runtime
    runtime = end_time - start_time

    return runtime

def save_runtime_info(model_path, runtime, output_file):
    """
    Save the runtime information to a JSON file.

    Parameters:
    model_path (str): The path to the ONNX model file.
    runtime (float): The runtime of the model in seconds.
    output_file (str): The path to the JSON file where the information will be saved.
    """
    # Create a dictionary to hold the information
    runtime_info = {
        "model_path": model_path,
        "runtime": runtime
    }

    # Save the information to a JSON file
    with open(output_file, 'w') as f:
            f.write(runtime_info + '\n')

if __name__ == "__main__":
    # Define the ONNX model path
    #model_path = "D:\\NAR\\NAR-Former-V2-main\\NAR-Former-V2-main\\dataset\\unseen_structure\\onnx\\nnmeter_mobilenetv2\\nnmeter_mobilenetv2_transform_0000.onnx"
    model_path = "D:\\NAR\\NAR-Former-V2-main\\NAR-Former-V2-main\\dataset5\\unseen_structure\\onnx\\mobilenetv2\\4.9714.onnx"
    # Generate random input data with shape 1x3x224x224
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Convert input data to the format required by the model
    input_data = {"input": input_data}

    # Test the runtime of the ONNX model
    runtime = test_onnx_model_runtime(model_path, input_data)

    # Define the output file path
    output_file = "runtime_info.json"

    # Save the runtime information
    save_runtime_info(model_path, runtime, output_file)

    print(f"Model runtime information saved to {output_file}")
