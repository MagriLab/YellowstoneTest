import os

import jax


def main(_) -> None:
    """
    ... we need some lines like this in the slurm file
    #SBATCH --gpus-per-task=8
    #SBATCH --cpus-per-gpu=10
    #SBATCH --ntasks=1

    srun python multi_node.py
    """

    jax.distributed.initialize(local_device_ids=range(8))

    print('jax.__version__', jax.__version__)

    print(os.environ.get('CUDA_VISIBLE_DEVICES'))

    print('jax.device_count()', jax.device_count())
    print('jax.local_device_count()', jax.local_device_count())
    print('jax.devices()', jax.devices())


if __name__ == '__main__':
    app.run(main)
