# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import math
from .topology import Topology, DistributedTopology, MachineTopology

def _get_remote_invbw(local_topology, opt_remote_invbw):
    if opt_remote_invbw is not None:
        if hasattr(local_topology, "remote_invbw"):
            print('Overriding remote_invbw value of local topology {} -> {}'.format(local_topology.remote_invbw,opt_remote_invbw))
        return opt_remote_invbw
    else:
        if not hasattr(local_topology, "remote_invbw"):
            assert False, "distributed topology expects remote bw to be provided"
        else:
            return local_topology.remote_invbw

def _copy_links(remote_lk, num_local, num_dist, local_links):
    return [[remote_lk if src // num_local != dst // num_local else local_links[dst % num_local][src % num_local]
        for src in range(num_dist)] for dst in range(num_dist)]

def _copy_invbws(remote_invbw, num_local, num_dist, local_invbws):
    return [[remote_invbw if src // num_local != dst // num_local else local_invbws[dst % num_local][src % num_local]
        for src in range(num_dist)] for dst in range(num_dist)]

def _copy_links_ext(remote_lk_func, num_local, num_dist, local_links):
    return [[remote_lk_func(src,dst) if src // num_local != dst // num_local else local_links[dst % num_local][src % num_local]
        for src in range(num_dist)] for dst in range(num_dist)]

def _copy_invbw_ext(remote_invbw_func, num_local, num_dist, local_invbws):
    return [[remote_invbw_func(src,dst) if src // num_local != dst // num_local else local_invbws[dst % num_local][src % num_local]
        for src in range(num_dist)] for dst in range(num_dist)]

# Keep number of switches (len(switches)) same, but add parallel instances of switches
def _copy_switches(num_local, num_copies, local_switches):
    switches = []
    for local_switch in local_switches:
        swt = []
        for i in range(num_copies):
            for srcs, dsts, lk, invbw, name in local_switch:
                dist_srcs = [src + i * num_local for src in srcs]
                dist_dsts = [dst + i * num_local for dst in dsts]
                swt.append((dist_srcs, dist_dsts, lk, invbw, f'copy_{i}_{name}_local'))
        switches.append(swt)
    return switches

# Add switches over IB
def _add_ext_switches(num_local, senders, receivers, remote_invbw):
    switches = []
    for sender in senders:
        others_recv = [other for other in receivers if other//num_local != sender//num_local]
        switches.append(([sender],others_recv,1,remote_invbw,f'node_{sender}_out'))
    for receiver in receivers:
        others_send = [other for other in senders if other//num_local != receiver//num_local]
        switches.append((others_send,[receiver],1,remote_invbw,f'node_{receiver}_in'))
    return switches

# Connects all external relay GPUs to each other with switches
# This prevents a GPU from receiving data from multiple GPUs over IB at the same time
# Supports only topologies which have defined alpha and beta costs
def distributed_relayed_switch(local_topology, num_copies,relays=[[0],[1]]):
    num_local = local_topology.num_nodes()
    num_dist = num_local * num_copies
    senders = [i*num_local + snd for i in range(num_copies) for snd in relays[0]]
    receivers = [i*num_local + rcv for i in range(num_copies) for rcv in relays[1]]
    links = _copy_links_ext(lambda src,dst: 1 if dst % num_local in relays[1] and src % num_local in relays[0] else 0,
        num_local, num_dist, local_topology.links)
    invbws = _copy_invbw_ext(lambda src,dst: local_topology.remote_invbw if dst % num_local in relays[1] and src % num_local in relays[0] else 0,
        num_local, num_dist, local_topology.invbws)
    switches = _copy_switches(num_local, num_copies, local_topology.switches)
    ext_switches = _add_ext_switches(num_local, senders, receivers, local_topology.remote_invbw)
    switches.append(ext_switches)
    return DistributedTopology(f'DistributedRelayedSwitch(local={local_topology.name},copies={num_copies})', num_copies, links, switches, invbws=invbws, remote_invbw=local_topology.remote_invbw, remote_alpha=local_topology.remote_alpha, remote_beta=local_topology.remote_beta, m_top=MachineTopology.RELAYED)

# All GPUs connected to every external GPU directly
# Does not use alpha-beta costs here
def distributed_fully_connected(local_topology, num_copies, opt_remote_invbw=None):
    num_local = local_topology.num_nodes()
    num_dist = num_local * num_copies
    remote_invbw = _get_remote_invbw(local_topology, opt_remote_invbw)

    links = _copy_links(1, num_local, num_dist, local_topology.links)
    invbws = _copy_invbws(remote_invbw, num_local, num_dist, local_topology.invbws)
    switches = _copy_switches(num_local, num_copies, local_topology.switches)

    return DistributedTopology(f'DistributedFullyConnected(local={local_topology.name},copies={num_copies})', num_copies, links, switches, invbws=invbws, remote_invbw=remote_invbw, remote_alpha=0, remote_beta=remote_invbw, m_top=MachineTopology.FULLY_CONNECTED)

# Relay GPUs connected to every external GPU directly
# Does not use alpha-beta costs here
def distributed_relayed(local_topology, num_copies, opt_remote_invbw=None,relays=[[0],[1]]):
    num_local = local_topology.num_nodes()
    num_dist = num_local * num_copies
    remote_invbw = _get_remote_invbw(local_topology, opt_remote_invbw)

    links = _copy_links_ext(lambda src,dst: 1 if dst % num_local in relays[1] and src % num_local in relays[0] else 0,
        num_local, num_dist, local_topology.links)
    invbws = _copy_invbw_ext(lambda src,dst: int(remote_invbw) if dst % num_local in relays[1] and src % num_local in relays[0] else 0,
        num_local, num_dist, local_topology.invbws)
    switches = _copy_switches(num_local, num_copies, local_topology.switches)

    return DistributedTopology(f'DistributedRelayed(local={local_topology.name},copies={num_copies},relays={relays})', num_copies, links, switches, invbws=invbws, remote_invbw=remote_invbw, remote_alpha=0, remote_beta=remote_invbw, m_top=MachineTopology.RELAYED)

# Connects all external relay GPUs to each other with switches
# Does not use alpha-beta costs here
def distributed_hub_and_spoke(local_topology, num_copies, opt_remote_invbw=None):
    num_local = local_topology.num_nodes()
    num_dist = num_local * num_copies
    remote_invbw = _get_remote_invbw(local_topology, opt_remote_invbw)

    links = _copy_links(remote_invbw, num_local, num_dist, local_topology.links)
    invbws = _copy_invbws(remote_invbw, num_local, num_dist, local_topology.invbws)
    switches = _copy_switches(num_local, num_copies, local_topology.switches)
    ext_switches = _add_ext_switches(num_local, list(range(num_local)), list(range(num_local)), remote_invbw)
    switches.append(ext_switches)
    
    return DistributedTopology(f'DistributedHubAndSpoke(local={local_topology.name},copies={num_copies})', num_copies, links, switches, invbws=invbws, remote_invbw=remote_invbw, remote_alpha=0, remote_beta=remote_invbw, m_top=MachineTopology.HUB_AND_SPOKE)
