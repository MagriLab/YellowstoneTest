# YellowstoneTest

This repo contains some sample code to test gpu capabilities on the Yellowstone cluster.

Ideally, we will be able to run the `multi_node.py` file to access gpus on multiple nodes. This should be tested with four nodes, eight gpus on each node.

The `jax.distributed.initialize` should not require the coordinator address or process id as the cluster uses the slurm scheduler. Some care must be taken to ensure than slurm launches a single process per gpu, as this is what jax expects.
