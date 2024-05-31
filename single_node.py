import os

import jax
from absl import app, flags

FLAGS = flags.FLAGS
_GPU = flags.DEFINE_boolean('gpu', False, 'Whether to test on GPU.')


def setup_cpu() -> None:
    flags = os.environ.get('XLA_FLAGS', '')
    flags += ' --xla_force_host_platform_device_count=8'

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['XLA_FLAGS'] = flags


def main(_) -> None:
    if not FLAGS.gpu:
        setup_cpu()

    print('jax.__version__', jax.__version__)

    if FLAGS.gpu:
        print(os.environ.get('CUDA_VISIBLE_DEVICES'))

    print('jax.device_count()', jax.device_count())
    print('jax.local_device_count()', jax.local_device_count())
    print('jax.devices()', jax.devices())


if __name__ == '__main__':
    app.run(main)
