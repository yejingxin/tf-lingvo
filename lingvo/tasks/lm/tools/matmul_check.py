# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Tool to decode GShard LM models.

Sample usage:

.. code-block:: bash

bazel run -c opt lingvo/tasks/lm/tools:matmul_check -- \
--tpu=yejingxin-tpu-v3 \
--disable_tf2=true

"""
import concurrent.futures
import functools
import sys
import time

from lingvo import compat as tf
from lingvo import model_registry
from lingvo.core import gshard_decode
from lingvo.core import py_utils
import numpy as np
from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import device_assignment as tpu_device_assignment
try:
  from tensorflow.python.compiler.xla.experimental import xla_sharding
except ImportError:
  # OSS backward compatibility, can be removed when TF is updated.
  from tensorflow.compiler.xla.experimental.xla_sharding import xla_sharding

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('tpu', '', 'TPU node address, if remote.')
tf.flags.DEFINE_boolean('disable_logging', False,
                        'disable all tf.logging calls below level '
                        'CRITICAL')
tf.flags.DEFINE_bool('disable_tf2', False,
                     'Whether run on Tensorflow without V2 behaviors.')
tf.flags.DEFINE_string(
    'computation_shape', '', 'Optionally restrict computation to a subset of '
    'available TPU cores. For example, set --computation_shape=4,8,1,2 to '
    'emulate DF 4x8 using DF 8x8 TPU job')

_daemon = gshard_decode.daemon


def override_flags():
  FLAGS.enable_asserts = False
  FLAGS.xla_device = 'tpu'

def main(unused_argv):
  override_flags()
  if FLAGS.disable_logging:
    tf.get_logger().setLevel('CRITICAL')
  cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
         FLAGS.tpu)
  _cluster_def = cluster_resolver.cluster_spec().as_cluster_def()
  _tpu = cluster_resolver.master()
  def _no_opt_sess_cfg():
    # Disable constant folding for convenience.
    return tf.config_pb2.ConfigProto(
        graph_options=tf.GraphOptions(
            optimizer_options=tf.OptimizerOptions(
                opt_level=tf.OptimizerOptions.L0,
                do_common_subexpression_elimination=False,
                do_function_inlining=False,
                do_constant_folding=False)),
        cluster_def=_cluster_def)
  sess = tf.Session(_tpu, config=_no_opt_sess_cfg())
  with sess.as_default(), sess.graph.as_default():
    tf.logging.info('Initializing TPU system')
    init_tpu = tf.tpu.initialize_system()
    topology_str = sess.run(init_tpu)
    topology = tf.tpu.experimental.Topology(topology_str)
    if FLAGS.computation_shape:
      computation_shape = list(map(int, FLAGS.computation_shape.split(',')))
    else:
      computation_shape = list(topology.mesh_shape)
    tf.logging.info('computation_shape: %r', computation_shape)
    tpu_cores = functools.reduce(lambda x, y: x * y, computation_shape)
    tf.logging.info('tpu_cores: %d', tpu_cores)
    device_assignment = tpu_device_assignment.device_assignment(
        topology_str, computation_shape=computation_shape, num_replicas=1)
    print('device assignmnet:', device_assignment)

    mesh_shape = [8, -1]
    device_mesh = np.arange(tpu_cores).reshape(mesh_shape)
    print('device_mesh:', device_mesh)
    print('mesh_shape:', device_mesh.shape)

    def tpu_fn():
      # x is a matrix [1024,8192] where all 1024 rows are the same
      x = tf.random.normal([8192],
                           mean=0.0,
                           stddev=1.0,
                           dtype=tf.bfloat16,
                           seed=10)
      x = tf.stack([x] * 1024, axis=0)
      assert x.shape == (1024, 8192)

      # x is sharded over mesh
      x = xla_sharding.mesh_split(
          x, device_mesh, [0, 1], use_sharding_op=True)

      # y is a random matrix
      y = tf.random.normal([8192, 65536],
                           mean=0.0,
                           stddev=1.0,
                           dtype=tf.bfloat16,
                           seed=20)

      # all rows in z should be the same
      return tf.einsum('ab,bc->ac', x, y)

    xla_op, run_op = tpu.split_compile_and_shard(
        computation=tpu_fn, num_shards=1, device_assignment=device_assignment)
    sess.run(xla_op)

    ret = sess.run(run_op)
    z = ret[0]
    tf.logging.info('row diff=%e', (z.min(axis=0) != z.max(axis=0)).mean())


if __name__ == '__main__':
  tf.profiler.experimental.start('/tmp/profile_dir')
  py_utils.SetEagerMode(False)
  FLAGS(sys.argv, known_only=True)
  if FLAGS.disable_tf2:
    tf.disable_v2_behavior()
  FLAGS.unparse_flags()
  tf.app.run(main)
  tf.profiler.experimental.stop()
