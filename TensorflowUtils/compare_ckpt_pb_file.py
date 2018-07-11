import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util

class CompLogFile(object):

    CKPT_DIR = 'D:/Source/PythonRepository/TensorflowUtils/files/model_graph'
    PB_DIR = 'D:/Source/PythonRepository/TensorflowUtils/files/freeze_graph.pb'

    def __init__(self):
        self.ckpt_dict = {}
        self.pb_dict = {}

        self._extract_ckpt()
        self._extract_pb()
        self._dict_to_set()

    def _extract_ckpt(self):
        '''
            학습된 ckpt 파일을 사용해 변수 명/값 추출
        '''
        print('>> Extracting CKPT File...')

        reader = pywrap_tensorflow.NewCheckpointReader(CompLogFile.CKPT_DIR)

        for key in reader.get_variable_to_shape_map():
            self.ckpt_dict[key] = reader.get_tensor(key)

    def _extract_pb(self):
        '''
            변환된 pb 파일을 사용해 변수 명/값 추출
        '''
        print('>> Extracting PB File...')

        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.3)
        )

        with tf.Session(config=config) as sess:
            with gfile.FastGFile(CompLogFile.PB_DIR, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')
                graph_nodes = [n for n in graph_def.node]

            wts = [n for n in graph_nodes if n.op == 'Const']

            for n in wts:
                self.pb_dict[n.name] = tensor_util.MakeNdarray(n.attr['value'].tensor)

    def _dict_to_set(self):
        print('>> Converting Dict to Set...')

        self.ckpt_set, self.pb_set = set(self.ckpt_dict.keys()), set(self.pb_dict.keys())
        self.intersect = self.ckpt_set.intersection(self.pb_set)

    def diff_set_ckpt(self):
        return self.ckpt_set - self.intersect

    def diff_set_pb(self):
        return self.pb_set - self.intersect

    def change_node(self):
        return set(n for n in self.intersect if (self.ckpt_dict[n] != self.pb_dict[n]).all())

    def unchange_node(self):
        return set(n for n in self.intersect if (self.ckpt_dict[n] == self.pb_dict[n]).all())

    def compare(self):
        with open('D:/Source/PythonRepository/TensorflowUtils/compare_result.txt', mode='wt') as f:
            f.write('>> Diff set CKPT File Nodes...\n')

            for node in self.diff_set_ckpt():
                f.write(node + '\n')

            f.write('\n\n>> Diff set PB File Nodes...\n')

            for node in self.diff_set_pb():
                f.write(node + '\n')

            f.write('\n\n>> Intersect Nodes...\n')

            for node in self.intersect:
                f.write(node + '\n')

            f.write('\n\n>> Changed Nodes...\n')

            for node in self.change_node():
                f.write(node + '\n')

            f.write('\n\n>> Unchanged Nodes...\n')

            for node in self.unchange_node():
                f.write(node + '\n')

comp = CompLogFile()
comp.compare()