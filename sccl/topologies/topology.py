# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from collections import defaultdict
from enum import Enum
import math

class Topology(object):
    def __init__(self, name, links, switches=[], invbws=None, remote_invbw=None, remote_alpha=None, remote_beta=None):
        self.name = name
        self.links = links
        self.invbws = links if invbws is None else invbws
        self.remote_invbw = remote_invbw
        self.remote_alpha = remote_alpha
        self.remote_beta = remote_beta
        self.switches = switches
        self.num_switches = len(switches)
        self.copies = 1
        for switch in switches:
            for srcs, dsts, lk, invbw, switch_name in switch:
                if lk == 0:
                    raise ValueError(f'Switch {switch_name} has zero bandwidth, but switch bandwidths must be strictly positive. Please encode connectedness in links.')
                if lk < 0:
                    raise ValueError(f'Switch {switch_name} has a negative inverse bandwidth of {invbw}. Bandwidth must be strictly positive.')
        self.bw_dist, _ = self.set_bw_distances()
        self.set_switches_involved()

    def sources(self, dst):
        for src, bw in enumerate(self.links[dst]):
            if bw > 0:
                yield src

    def destinations(self, src):
        for dst, links in enumerate(self.links):
            bw = links[src]
            if bw > 0:
                yield dst

    def link(self, src, dst):
        return self.links[dst][src]

    def get_invbw(self, src, dst):
        return self.invbws[dst][src]

    def num_nodes(self):
        return len(self.links)

    def nodes(self):
        return range(self.num_nodes())

    # constraints using number of links
    def bandwidth_constraints(self):
        for dst, dst_links in enumerate(self.links):
            for src, lk in enumerate(dst_links):
                if lk > 0:
                    yield ([src], [dst], lk, f'{src}→{dst}')
        for switch in self.switches:
            for srcs, dsts,lk, _, switch_name in switch:
                yield (srcs, dsts, lk, switch_name)

    # constraints using actual bandwidth
    def real_bandwidth_constraints(self):
        for dst, dst_links in enumerate(self.invbws):
            for src, invbw in enumerate(dst_links):
                if invbw > 0:
                    yield ([src], [dst], invbw, f'{src}→{dst}')
        for switch in self.switches:
            for srcs, dsts, _, invbw, switch_name in switch:
                yield (srcs, dsts, invbw, switch_name)

    def set_bw_distances(self):
        # Floyd–Warshall algorithm for all-pairs shortest paths with path information
        # Modified to track all shortest paths
        nodes = range(self.num_nodes())
        dist = [[math.inf for _ in nodes] for _ in nodes]
        next = [[set() for _ in nodes] for _ in nodes]
        for dst in nodes:
            for src in self.sources(dst):
                dist[src][dst] = self.invbws[dst][src]
                next[src][dst].add(dst)
        for node in nodes:
            dist[node][node] = 0
            next[node][node].add(node)
        for k in nodes:
            for i in nodes:
                for j in nodes:
                    if dist[i][j] >= dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next[i][j].update(next[i][k])
        return dist, next

    def set_switches_involved(self):
        self.switches_involved = defaultdict(list)
        for i, switch in enumerate(self.switches):
            for srcs, dsts, _, _, switch_name in switch:
                if "out" in switch_name:
                    for src in srcs:
                        for dst in dsts:
                            self.switches_involved[(src,dst)].append(i)


class MachineTopology(Enum):
    FULLY_CONNECTED = 'FULLY_CONNECTED'
    HUB_AND_SPOKE = 'HUB_AND_SPOKE'
    RELAYED = 'RELAYED'

    def __str__(self):
        return self.value

class DistributedTopology(Topology):
    def __init__(self, name, copies, links, switches=[], invbws=None, remote_invbw=None, remote_alpha=None, remote_beta=None, m_top=MachineTopology.RELAYED, relay=[[0],[1]]):
        super().__init__(name, links, switches, invbws=invbws, remote_invbw=remote_invbw, remote_alpha=remote_alpha, remote_beta=remote_beta)
        self.copies = copies
        self.m_top = m_top
        self.relay = relay if m_top == MachineTopology.RELAYED else None
