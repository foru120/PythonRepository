import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util

GRAPH_PB_PATH = 'D:/Source/PythonRepository/TensorflowUtils/files/saved_model.pb'

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.7)
)

with tf.Session(config=config) as sess:
    print('load graph')
    with gfile.FastGFile(GRAPH_PB_PATH, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        graph_nodes = [n for n in graph_def.node]

    wts = [n for n in graph_nodes if n.op == 'Const']

    for n in wts:
        # if n.name == 'right/output_network/batch_norm_output/moving_var':
        print('Name of the node - %s' % (n.name))
        print('Value - ')
        print(tensor_util.MakeNdarray(n.attr['value'].tensor))