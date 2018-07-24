import tensorflow as tf
import tensorflow.contrib.slim as slim
from Hongbog.DementiaDiagnosis.wt_mobilenet import mobilenet_v1, training_scope


class MobilenetV1Test(tf.test.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def arg_scope_func_key(self, op):
        return getattr(op, '_key_op', str(op))

    def testBatchNormScopeDoesNotHaveIsTrainingItsSetToNone(self):
        sc = training_scope(is_training=None)
        self.assertNotIn('is_training', sc[self.arg_scope_func_key(slim.batch_norm)])

    def testBatchNormScopeDoesHasIsTrainingWhenItsNotNone(self):
        sc = training_scope(is_training=False)
        self.assertIn('is_training', sc[self.arg_scope_func_key(slim.batch_norm)])
        sc = training_scope(is_training=True)
        self.assertIn('is_training', sc[self.arg_scope_func_key(slim.batch_norm)])
        sc = training_scope()
        self.assertIn('is_training', sc[self.arg_scope_func_key(slim.batch_norm)])

if __name__ == '__main__':
    tf.test.main()