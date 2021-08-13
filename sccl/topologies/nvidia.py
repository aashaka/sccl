# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .topology import Topology

from fractions import Fraction
import subprocess

# DGX1 config assuming 1MB data transfer size
def dgx1():
    # (0 1 2 3) (4 5 6 7) are two sockets
    # 0 1 3 2 is the high bandwidth chain in socket 1
    # 4 5 7 6 is the high bandwidth chain in socket 2
    # 0 4 and 2 6 are high bandwidth intersocket links

    # links = [
    #     #0  1  2  3  4  5  6  7
    #     [0, 2, 1, 1, 2, 0, 0, 0],
    #     [2, 0, 1, 2, 0, 1, 0, 0],
    #     [1, 1, 0, 2, 0, 0, 2, 0],
    #     [1, 2, 2, 0, 0, 0, 0, 1],
    #     [2, 0, 0, 0, 0, 2, 1, 1],
    #     [0, 1, 0, 0, 2, 0, 1, 2],
    #     [0, 0, 2, 0, 1, 1, 0, 2],
    #     [0, 0, 0, 1, 1, 2, 2, 0]
    # ]

    # Link connection matrix
    links = [
        #0  1  2  3  4  5  6  7
        [0, 1, 1, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 1, 1, 1, 0]
    ]

    # NVLink bandwidth for each link
    # high bandwidth link => alpha=0, beta=23/46
    invbws = [
        [0, 23, 46, 46, 23, 0, 0, 0],
        [23, 0, 46, 23, 0, 46, 0, 0],
        [46, 46, 0, 23, 0, 0, 23, 0],
        [46, 23, 23, 0, 0, 0, 0, 46],
        [23, 0, 0, 0, 0, 23, 46, 46],
        [0, 46, 0, 0, 23, 0, 46, 23],
        [0, 0, 23, 0, 46, 46, 0, 23],
        [0, 0, 0, 46, 46, 23, 23, 0]
    ]

    remote_invbw = 107  # approx IB alpha + beta = 2 + 105
    remote_alpha = 2.6  # IB alpha = 2.6
    remote_beta = 105   # IB beta = 105

    # self.symmetries = [
    #     [0, 1, 2, 3, 4, 5, 6, 7], #0 goes to itself
    #     [0, 1, 2, 3, 4, 5, 6, 7], #1 goes to itself
    #     [2, 3, 0, 1, 6, 7, 4, 5], #2 goes to 0, 3 goes to 1, ... top - bottom symmetry
    #     [2, 3, 0, 1, 6, 7, 4, 5], #3 goes to 1, 2 goes to 0, ... top - bottom symmetry
    #     [4, 5, 6, 7, 0, 1, 2, 3], #4 goes to 0, 5 goes to 1, ... left - right symmetry
    #     [4, 5, 6, 7, 0, 1, 2, 3], #5 goes to 1, 4 goes to 0, ... left - right symmetry
    #     [6, 7, 4, 5, 2, 3, 0, 1], #6 goes to 0, 7 goes to 1, ... top-bottom + left-right
    #     [6, 7, 4, 5, 2, 3, 0, 1]  #7 goes to 1, 6 goes to 0, ... top-bottom + left-right
    # ]

    # self.beta_bound = Fraction(7,6)
    # self.diameter = 2

    return Topology('DGX1', links, invbws=invbws, remote_invbw=remote_invbw, remote_alpha=remote_alpha, remote_beta=remote_beta)

# DGX1 config with high link latency
def dgx1_lat():
    # (0 1 2 3) (4 5 6 7) are two sockets
    # 0 1 3 2 is the high bandwidth chain in socket 1
    # 4 5 7 6 is the high bandwidth chain in socket 2
    # 0 4 and 2 6 are high bandwidth intersocket links

    # Link connection matrix
    links = [
        #0  1  2  3  4  5  6  7
        [0, 1, 1, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 1, 1, 1, 0]
    ]

    # NVLink bandwidth for each link
    # high latency link => alpha=0.3, beta=0
    invbws = [
        [0, 0.3, 0.3, 0.3, 0.3, 0, 0, 0],
        [0.3, 0, 0.3, 0.3, 0, 0.3, 0, 0],
        [0.3, 0.3, 0, 0.3, 0, 0, 0.3, 0],
        [0.3, 0.3, 0.3, 0, 0, 0, 0, 0.3],
        [0.3, 0, 0, 0, 0, 0.3, 0.3, 0.3],
        [0, 0.3, 0, 0, 0.3, 0, 0.3, 0.3],
        [0, 0, 0.3, 0, 0.3, 0.3, 0, 0.3],
        [0, 0, 0, 0.3, 0.3, 0.3, 0.3, 0]
    ]

    remote_invbw = 3    # IB alpha + beta = 3 + 0
    remote_alpha = 3    # IB alpha = 3
    remote_beta = 0     # IB beta = 0

    return Topology('DGX1Lat', links, invbws=invbws, remote_invbw=remote_invbw, remote_alpha=remote_alpha, remote_beta=remote_beta)

def nvlink_only(nvidia_smi_topo=None):
    if nvidia_smi_topo == None:
        nvidia_smi_topo = _get_nvidia_smi_topo()
    links = _parse_nvidia_smi_topo(nvidia_smi_topo)
    return Topology('NVLinkOnly', links)

def _get_nvidia_smi_topo():
    output = subprocess.check_output("nvidia-smi topo -m".split())
    return output.decode("utf-8")

def _parse_nvidia_smi_topo(output):
    lines = output.splitlines()
    before_legend = []
    for l in lines[1:]:
        if l and l.startswith("GPU"):
            # Only look at the rows for GPU
            before_legend.append(l)
        else:
            break
    devices = [x.split("\t")[0] for x in before_legend]
    gpus = [i for i in range(len(before_legend))
            if before_legend[i].startswith("GPU")]
    matrix = [x.split("\t")[1:] for x in before_legend]
    nvlink_matrix = [[_nvlink_num(x[g]) for g in gpus] for x in matrix]
    return nvlink_matrix

def _nvlink_num(x):
    x = x.strip()
    if x.startswith("NV"):
        return int(x[2:])
    else:
        return 0
