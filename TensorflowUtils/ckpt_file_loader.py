from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

CKPT_FILE_DIR = 'D:/Source/PythonRepository/TensorflowUtils/files/model_graph'

print_tensors_in_checkpoint_file(file_name=CKPT_FILE_DIR, tensor_name='', all_tensors=True)