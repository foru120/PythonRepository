'''
▣ 스크립트 사용 방법
python freeze_graph.py \
--input_graph=/home/hwang/work/flowers/train/graph.pbtxt \
--input_checkpoint=/home/hwang/work/flowers/train/model.ckpt-15000 \
--output_graph=/home/hwang/work/flowers/freeze.pb \
--output_node_names=InceptionResnetV2/Logits/Predictions

--input_grap= 위에서 준비한 체크포인트 파일 경로 (그래프 정보)
--input_checkpoint= 위에서 준비한 체크포인트 파일 경로 (체크포인트 파일)
--output_graph= 출력 할 pb 파일 경로 및 이름 설정
--output_node_names= 필요한 그래프 노드
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from Study.google import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.platform import app
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import saved_model_cli
from tensorflow.python.training import saver as saver_lib

def freeze_graph_with_def_protos(input_graph_def,
                                 input_saver_def,
                                 input_checkpoint,
                                 output_node_names,
                                 restore_op_name,
                                 filename_tensor_name,
                                 output_graph,
                                 clear_devices,
                                 initializer_nodes,
                                 variable_names_whitelist="",
                                 variable_names_blacklist="",
                                 input_meta_graph_def=None,
                                 input_saved_model_dir=None,
                                 saved_model_tags=None,
                                 checkpoint_version=saver_pb2.SaverDef.V2):
    """Converts all variables in a graph and checkpoint into constants."""
    del restore_op_name, filename_tensor_name  # Unused by updated loading code.

    # 'input_checkpoint' may be a prefix if we're using Saver V2 format
    if (not input_saved_model_dir and
            not saver_lib.checkpoint_exists(input_checkpoint)):
        print("Input checkpoint '" + input_checkpoint + "' doesn't exist!")
        return -1

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # Remove all the explicit device specifications for this node. This helps to
    # make the graph more portable.
    if clear_devices:
        if input_meta_graph_def:
            for node in input_meta_graph_def.graph_def.node:
                node.device = ""
        elif input_graph_def:
            for node in input_graph_def.node:
                node.device = ""

    if input_graph_def:
        _ = importer.import_graph_def(input_graph_def, name="")
    with session.Session() as sess:
        if input_saver_def:
            saver = saver_lib.Saver(
                saver_def=input_saver_def, write_version=checkpoint_version)
            saver.restore(sess, input_checkpoint)
        elif input_meta_graph_def:
            restorer = saver_lib.import_meta_graph(
                input_meta_graph_def, clear_devices=True)
            restorer.restore(sess, input_checkpoint)
            if initializer_nodes:
                sess.run(initializer_nodes.replace(" ", "").split(","))
        elif input_saved_model_dir:
            if saved_model_tags is None:
                saved_model_tags = []
            loader.load(sess, saved_model_tags, input_saved_model_dir)
        else:
            var_list = {}
            reader = pywrap_tensorflow.NewCheckpointReader(input_checkpoint)
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in var_to_shape_map:
                try:
                    tensor = sess.graph.get_tensor_by_name(key + ":0")
                except KeyError:
                    # This tensor doesn't exist in the graph (for example it's
                    # 'global_step' or a similar housekeeping element) so skip it.
                    continue
                var_list[key] = tensor
            saver = saver_lib.Saver(
                var_list=var_list, write_version=checkpoint_version)
            saver.restore(sess, input_checkpoint)
            if initializer_nodes:
                sess.run(initializer_nodes.replace(" ", "").split(","))

        variable_names_whitelist = (
            variable_names_whitelist.replace(" ", "").split(",")
            if variable_names_whitelist else None)
        variable_names_blacklist = (
            variable_names_blacklist.replace(" ", "").split(",")
            if variable_names_blacklist else None)

        if input_meta_graph_def:
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                input_meta_graph_def.graph_def,
                output_node_names.replace(" ", "").split(","),
                variable_names_whitelist=variable_names_whitelist,
                variable_names_blacklist=variable_names_blacklist)
        else:
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                output_node_names.replace(" ", "").split(","),
                variable_names_whitelist=variable_names_whitelist,
                variable_names_blacklist=variable_names_blacklist)

    # Write GraphDef to file if output path has been given.
    if output_graph:
        with gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

    return output_graph_def

def _parse_input_graph_proto(input_graph, input_binary):
    """Parser input tensorflow graph into GraphDef proto."""
    if not gfile.Exists(input_graph):
        print("Input graph file '" + input_graph + "' does not exist!")
        return -1
    input_graph_def = graph_pb2.GraphDef()
    mode = "rb" if input_binary else "r"
    with gfile.FastGFile(input_graph, mode) as f:
        if input_binary:
            input_graph_def.ParseFromString(f.read())
        else:
            text_format.Merge(f.read(), input_graph_def)
    return input_graph_def

def _parse_input_meta_graph_proto(input_graph, input_binary):
    """Parser input tensorflow graph into MetaGraphDef proto."""
    if not gfile.Exists(input_graph):
        print("Input meta graph file '" + input_graph + "' does not exist!")
        return -1
    input_meta_graph_def = MetaGraphDef()
    mode = "rb" if input_binary else "r"
    with gfile.FastGFile(input_graph, mode) as f:
        if input_binary:
            input_meta_graph_def.ParseFromString(f.read())
        else:
            text_format.Merge(f.read(), input_meta_graph_def)
    print("Loaded meta graph file '" + input_graph)
    return input_meta_graph_def

def _parse_input_saver_proto(input_saver, input_binary):
    """Parser input tensorflow Saver into SaverDef proto."""
    if not gfile.Exists(input_saver):
        print("Input saver file '" + input_saver + "' does not exist!")
        return -1
    mode = "rb" if input_binary else "r"
    with gfile.FastGFile(input_saver, mode) as f:
        saver_def = saver_pb2.SaverDef()
        if input_binary:
            saver_def.ParseFromString(f.read())
        else:
            text_format.Merge(f.read(), saver_def)
    return saver_def

def freeze_graph(input_graph,
                 input_saver,
                 input_binary,
                 input_checkpoint,
                 output_node_names,
                 restore_op_name,
                 filename_tensor_name,
                 output_graph,
                 clear_devices,
                 initializer_nodes,
                 variable_names_whitelist="",
                 variable_names_blacklist="",
                 input_meta_graph=None,
                 input_saved_model_dir=None,
                 saved_model_tags=tag_constants.SERVING,
                 checkpoint_version=saver_pb2.SaverDef.V2):
    """Converts all variables in a graph and checkpoint into constants."""
    input_graph_def = None
    if input_saved_model_dir:
        input_graph_def = saved_model_cli.get_meta_graph_def(
            input_saved_model_dir, saved_model_tags).graph_def
    elif input_graph:
        input_graph_def = _parse_input_graph_proto(input_graph, input_binary)
    input_meta_graph_def = None
    if input_meta_graph:
        input_meta_graph_def = _parse_input_meta_graph_proto(
            input_meta_graph, input_binary)
    input_saver_def = None
    if input_saver:
        input_saver_def = _parse_input_saver_proto(input_saver, input_binary)
    freeze_graph_with_def_protos(
        input_graph_def,
        input_saver_def,
        input_checkpoint,
        output_node_names,
        restore_op_name,
        filename_tensor_name,
        output_graph,
        clear_devices,
        initializer_nodes,
        variable_names_whitelist,
        variable_names_blacklist,
        input_meta_graph_def,
        input_saved_model_dir,
        saved_model_tags.replace(" ", "").split(","),
        checkpoint_version=checkpoint_version)

def main(unused_args, flags):
    if flags.checkpoint_version == 1:
        checkpoint_version = saver_pb2.SaverDef.V1
    elif flags.checkpoint_version == 2:
        checkpoint_version = saver_pb2.SaverDef.V2
    else:
        print("Invalid checkpoint version (must be '1' or '2'): %d" %
              flags.checkpoint_version)
        return -1
    freeze_graph(flags.input_graph, flags.input_saver, flags.input_binary,
                 flags.input_checkpoint, flags.output_node_names,
                 flags.restore_op_name, flags.filename_tensor_name,
                 flags.output_graph, flags.clear_devices, flags.initializer_nodes,
                 flags.variable_names_whitelist, flags.variable_names_blacklist,
                 flags.input_meta_graph, flags.input_saved_model_dir,
                 flags.saved_model_tags, checkpoint_version)

def run_main():
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--input_graph",
        type=str,
        default="D:/05_source/PythonRepository/Hongbog/EyeVerification/train_log/4_1th_test/graph.pbtxt",
        help="TensorFlow \'GraphDef\' file to load.")
    parser.add_argument(
        "--input_saver",
        type=str,
        default="",
        help="TensorFlow saver file to load.")
    parser.add_argument(
        "--input_checkpoint",
        type=str,
        default="D:/05_source/PythonRepository/Hongbog/EyeVerification/train_log/4_1th_test/eye_verification_param-56.data-00000-of-00001",
        help="TensorFlow variables file to load.")
    parser.add_argument(
        "--checkpoint_version",
        type=int,
        default=2,
        help="Tensorflow variable file format")
    parser.add_argument(
        "--output_graph",
        type=str,
        default="D:/05_source/PythonRepository/Hongbog/EyeVerification/train_log/4_1th_test/eye_verification.pb",
        help="Output \'GraphDef\' file name.")
    parser.add_argument(
        "--input_binary",
        nargs="?",
        const=True,
        type="bool",
        default=False,
        help="Whether the input files are in binary format.")
    parser.add_argument(
        "--output_node_names",
        type=str,
        default="softmax,softmax_1",
        help="The name of the output nodes, comma separated.")
    parser.add_argument(
        "--restore_op_name",
        type=str,
        default="save/restore_all",
        help="""\
      The name of the master restore operator. Deprecated, unused by updated \
      loading code.
      """)
    parser.add_argument(
        "--filename_tensor_name",
        type=str,
        default="save/Const:0",
        help="""\
      The name of the tensor holding the save path. Deprecated, unused by \
      updated loading code.
      """)
    parser.add_argument(
        "--clear_devices",
        nargs="?",
        const=True,
        type="bool",
        default=True,
        help="Whether to remove device specifications.")
    parser.add_argument(
        "--initializer_nodes",
        type=str,
        default="",
        help="Comma separated list of initializer nodes to run before freezing.")
    parser.add_argument(
        "--variable_names_whitelist",
        type=str,
        default="",
        help="""\
      Comma separated list of variables to convert to constants. If specified, \
      only those variables will be converted to constants.\
      """)
    parser.add_argument(
        "--variable_names_blacklist",
        type=str,
        default="",
        help="""\
      Comma separated list of variables to skip converting to constants.\
      """)
    parser.add_argument(
        "--input_meta_graph",
        type=str,
        default="",
        help="TensorFlow \'MetaGraphDef\' file to load.")
    parser.add_argument(
        "--input_saved_model_dir",
        type=str,
        default="",
        help="Path to the dir with TensorFlow \'SavedModel\' file and variables.")
    parser.add_argument(
        "--saved_model_tags",
        type=str,
        default="serve",
        help="""\
      Group of tag(s) of the MetaGraphDef to load, in string format,\
      separated by \',\'. For tag-set contains multiple tags, all tags \
      must be passed in.\
      """)
    flags, unparsed = parser.parse_known_args()

    my_main = lambda unused_args: main(unused_args, flags)
    app.run(main=my_main, argv=[sys.argv[0]] + unparsed)

if __name__ == '__main__':
    run_main()