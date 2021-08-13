# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sccl.collectives import *
from sccl.algorithm import *
from sccl.instance import *
from sccl.topologies import distributed_fully_connected, distributed_relayed

def synthesize_allgather_bcast_distributed_allgather(num_copies, gather_algo, scatter_algo, logging=False):
    print("gather_algo chunks: ", gather_algo.instance.chunks)
    print("bcast_algo chunks: ", scatter_algo.instance.chunks)
    if gather_algo.is_pipelined() or scatter_algo.is_pipelined():
        raise ValueError('Pipelining is not supported.')

    # if gather_algo.instance.chunks != scatter_algo.instance.chunks:
    #     raise ValueError(f'Local gather and local scatter must have the same chunks (got {gather_algo.instance.chunks} and {scatter_algo.instance.chunks})')

    if gather_algo.topology.name != scatter_algo.topology.name:
        # TODO: improve this check to check actual structure, not just name
        raise ValueError(f'Local gather and local scatter must have the same topology (got {gather_algo.topology.name} and {scatter_algo.topology.name})')
    local_topology = gather_algo.topology

    chunks = gather_algo.instance.chunks
    chunks_bcast = scatter_algo.instance.chunks
    local_nodes = gather_algo.topology.num_nodes()
    nodes = local_nodes * num_copies

    steps = []

    for local_step in gather_algo.steps:
        sends = []

        # Translate copies of the local AllGather to the new space of ranks
        for chunk, src, dst in local_step.sends:
            for i in range(num_copies):
                src_copy = src + i * local_nodes
                chunk_copy = chunk + i * chunks * local_nodes
                dst_copy = dst + i * local_nodes

                # Translate send src and dst to distributed space and the send to the distributed algorithm
                sends.append((chunk_copy, src_copy, dst_copy))
                assert src_copy != dst_copy

        steps.append(Step(local_step.rounds, sends))

    # Perform transpose between local root nodes
    transpose_sends = []
    for i in range(num_copies):
        for j in range(num_copies):
            for c in range(chunks * local_nodes):
                if i == j:
                    continue
                src_copy = i * local_nodes
                dst_copy = j * local_nodes
                chunk_copy = c + i * chunks * local_nodes
                transpose_sends.append((chunk_copy, src_copy, dst_copy))

    steps.append(Step(chunks*local_nodes, transpose_sends))

    div_factor = (chunks * num_copies * local_nodes) // chunks_bcast
    for local_step in scatter_algo.steps:
        sends = []

        # Translate copies of the local Scatter to the new space of ranks
        for i in range(num_copies):
            for chunk, src, dst in local_step.sends:
                src_copy = src + i * local_nodes
                dst_copy = dst + i * local_nodes
                for chunk_copy in range(chunks * local_nodes * num_copies):
                    if (chunk_copy // (chunks*local_nodes)) == i:
                        continue
                    if chunk_copy // div_factor != chunk:
                        continue
                    # Translate send src and dst to distributed space and the send to the distributed algorithm
                    sends.append((chunk_copy, src_copy, dst_copy))
        steps.append(Step(local_step.rounds * chunks * local_nodes * (num_copies-1), sends))

    collective = allgather(nodes)
    topology = distributed_relayed(gather_algo.topology, num_copies, 1)

    instance = Instance(
        steps=len(steps),
        extra_rounds=sum(step.rounds - 1 for step in steps),
        chunks=chunks,
    )
    return Algorithm.make_implementation(collective, topology, instance, steps)