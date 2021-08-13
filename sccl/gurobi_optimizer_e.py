# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sccl import topologies
from sccl.algorithm import *
from sccl.instance import *
from sccl.shortest_path_sets import *
from gurobipy import GRB, Model, quicksum
from sccl.topologies.topology import DistributedTopology, MachineTopology
import numpy as np

class GurobiOptimizerE(object):
    def __init__(self, topology, collective):
        self.topology = topology
        self.collective_og = collective

    def _is_relay_link(self,r,dst):
        if isinstance(self.topology, DistributedTopology) and self.topology.m_top == MachineTopology.RELAYED:
            copies = self.topology.copies
            num_local_nodes = self.topology.num_nodes() // copies
            if r % num_local_nodes in self.topology.relay[0] and dst % num_local_nodes in self.topology.relay[1] and r // num_local_nodes != dst // num_local_nodes:
                return True

    def _encode(self, opt, chunk_order, chunk_time,
        switch_chunk_order_recv, switch_chunk_time_recv, switch_chunk_order_send, switch_chunk_time_send,
        nic_chunk_order_recv, nic_chunk_time_recv, nic_chunk_order_send, nic_chunk_time_send,
        heuristic=3, extra_heuristic=True):

        self.spsets = shortest_path_sets(self.topology, self.collective)
        C = self.collective.num_chunks
        R = self.collective.num_nodes
        smallM = 10
        M = 10000000 # big-M for maximum self.time between sends
        ST = 5000000 # self.time for unsent sends and unstarted starts
        SND = 10000000 # self.time for unsent sends and unstarted starts
        opt.Params.Threads = 4

        L = 0
        for src in range(R):
            for dst in self.topology.destinations(src):
                if self.topology.link(src,dst) > L:
                    L = self.topology.link(src,dst)

        self.is_sent_set_1 = set()
        self.is_before_set_1 = set()
        self.is_together_set_0 = set()
        self.is_together_set_1 = set()
        self.recv_first_set_1 = set()
        self.nic_recv_first_set_1 = set()
        self.send_first_set_1 = set()
        self.nic_send_first_set_1 = set()

        self.is_before = {}
        self.is_together = {}
        self.recv_first = {}
        self.send_first = {}

        self.send = opt.addVars(C, R, R, L, name="send", vtype=GRB.CONTINUOUS, lb=0.0)
        self.start = opt.addVars(C, R, name="start", vtype=GRB.CONTINUOUS, lb=0.0)
        self.time = opt.addVar(name="time", vtype=GRB.CONTINUOUS)


        opt.setObjective(self.time, GRB.MINIMIZE)
        opt.addLConstr(self.time <= ST-1)

        def minmax(c,o):
            if c <= o:
                return (c,o)
            else:
                return (o,c)

        #  Only try together over relay
        def _should_try_together(r,dst,c,o):
            # return False
            if not (isinstance(self.topology, DistributedTopology) and self.topology.m_top == MachineTopology.RELAYED):
                return False
            for rc in self.collective.pre_on(c):
                r1 = rc
            for ro in self.collective.pre_on(o):
                r2 = ro
            assert r != dst
            if self._is_relay_link(r,dst):
                # if self.topology.bw_dist[rc][r] == self.topology.bw_dist[ro][r]:
                return True
            return False

        def _should_fix_together(r,dst,c,o):
            return False
            if not (isinstance(self.topology, DistributedTopology) and self.topology.m_top == MachineTopology.RELAYED):
                return False
            for rc in self.collective.pre_on(c):
                r1 = rc
            for ro in self.collective.pre_on(o):
                r2 = ro
            assert r != dst
            if self._is_relay_link(r,dst):
                if self.topology.bw_dist[rc][r] == self.topology.bw_dist[ro][r]:
                    # print("Set true", c,o,r)
                    return True
                # if self.topology.links[r][ro] == 0 and self.topology.links[r][rc] == 0:
                #     print("Sending 1")
                #     return True
                # if self.topology.links[r][ro] != 0 and self.topology.links[r][rc] != 0:
                #     print("Sending 2")
                #     return True
            return False

        def _add_chunk_sent(opt, heuristic):
            if chunk_order is not None:
                assert(chunk_time is not None)
                assert(len(chunk_order) == R)
                assert(len(chunk_order[0]) == R)
                for r in range(R):
                    for src in self.topology.sources(r):
                        lchunks = [[] for l in range(self.topology.link(src,r))]
                        numl = self.topology.link(src,r)
                        if heuristic == 3 or heuristic == 5:
                            i=0
                            for c in chunk_order[r][src][0]:
                                lchunks[i].append(c)
                                i = (i+1)%numl
                        else:
                            for l in range(numl):
                                lchunks[l] = chunk_order[r][src][l]
                        for l in range(numl):
                            i = 0
                            while i < len(lchunks[l]):
                                c = lchunks[l][i]
                                if r in self.spsets[c]:
                                    self.is_sent_set_1.add((c,src,r,l))
                                i = i + 1

        def _add_nic_order(opt):
            recv_right_after = [defaultdict() for r in range(R//2)]
            send_right_after = [defaultdict() for r in range(R//2)]
            if nic_chunk_order_recv is not None:
                assert(nic_chunk_time_recv is not None)
                assert(len(nic_chunk_order_recv) == R//2)
                for ni in range(R//2):
                    lchunks = nic_chunk_order_recv[ni]
                    for l in range(len(lchunks)):
                        i = 0
                        while i < len(lchunks[l]):
                            j = i + 1
                            c, srci, ri = lchunks[l][i]
                            has_after = False
                            while j<len(lchunks[l]):
                                o, srcj, rj = lchunks[l][j]
                                if srci != srcj and ri != rj:   # the case where srci != srcj but ri == rj will be handled in switch ordering, both equals will be handled in chunk_ordering
                                    self.nic_recv_first_set_1.add((c,o,ri,srci,rj,srcj))
                                    if not has_after:
                                        recv_right_after[ni][(c,srci,ri)] = (o,srcj,rj)
                                        has_after = True
                                j = j + 1
                            if not has_after:
                                recv_right_after[ni][(c,srci,ri)] = (-1,-1,-1)
                            i = i + 1
                    lchunks = nic_chunk_order_send[ni]
                    for l in range(len(lchunks)):
                        i = 0
                        while i < len(lchunks[l]):
                            j = i + 1
                            c, dsti, ri = lchunks[l][i]
                            has_after = False
                            while j<len(lchunks[l]):
                                o, dstj, rj = lchunks[l][j]
                                if dsti != dstj and ri != rj:
                                    self.nic_send_first_set_1.add((c,o,ri,dsti,rj,dstj))
                                    if not has_after:
                                        send_right_after[ni][(c,dsti,ri)] = (o,dstj,rj)
                                        has_after = True
                                j = j + 1
                            if not has_after:
                                send_right_after[ni][(c,dsti,ri)] = (-1,-1,-1)
                            i = i + 1
            return recv_right_after, send_right_after

        def _add_switch_order(opt):
            recv_right_after = [defaultdict() for r in range(R)]
            send_right_after = [defaultdict() for r in range(R)]
            if switch_chunk_order_recv is not None:
                assert(switch_chunk_time_recv is not None)
                assert(len(switch_chunk_order_recv) == R)
                for r in range(R):
                    lchunks = switch_chunk_order_recv[r]
                    for l in range(len(lchunks)):
                        i = 0
                        while i < len(lchunks[l]):
                            j = i + 1
                            c, srci = lchunks[l][i]
                            has_after = False
                            while j<len(lchunks[l]):
                                o, srcj = lchunks[l][j]
                                if srci != srcj:
                                    self.recv_first_set_1.add((c,o,r,srci,srcj))
                                    if not has_after:
                                        recv_right_after[r][(c,srci)] = (o,srcj)
                                        has_after = True
                                j = j + 1
                            if not has_after:
                                recv_right_after[r][(c,srci)] = (-1,-1)
                            i = i + 1
                    lchunks = switch_chunk_order_send[r]
                    for l in range(len(lchunks)):
                        i = 0
                        while i < len(lchunks[l]):
                            j = i + 1
                            c, dsti = lchunks[l][i]
                            has_after = False
                            while j<len(lchunks[l]):
                                o, dstj = lchunks[l][j]
                                if dsti != dstj:
                                    self.send_first_set_1.add((c,o,r,dsti,dstj))
                                    if not has_after:
                                        send_right_after[r][(c,dsti)] = (o,dstj)
                                        has_after = True
                                j = j + 1
                            if not has_after:
                                send_right_after[r][(c,dsti)] = (-1,-1)
                            i = i + 1
            return recv_right_after, send_right_after

        def _add_chunk_order_dgx2(opt, heuristic, recv_right_after, send_right_after):
            if chunk_order is not None:
                assert(chunk_time is not None)
                assert(len(chunk_order) == R)
                assert(len(chunk_order[0]) == R)
                for r in range(R):
                    for src in self.topology.sources(r):
                        lchunks = [[] for l in range(self.topology.link(src,r))]
                        numl = self.topology.link(src,r)
                        if heuristic == 3 or heuristic == 5:
                            i=0
                            for c in chunk_order[r][src][0]:
                                lchunks[i].append(c)
                                i = (i+1)%numl
                        else:
                            for l in range(numl):
                                lchunks[l] = chunk_order[r][src][l]
                        for l in range(numl):
                            i = 0
                            while i < len(lchunks[l]):
                                j = i + 1
                                c = lchunks[l][i]
                                while j<len(lchunks[l]):
                                    o = lchunks[l][j]
                                    c1,o1 = minmax(c,o)
                                    prev_o = lchunks[l][j-1]
                                    c2, prev_o2 = minmax(c,prev_o)
                                    if r in self.spsets[c] and r in self.spsets[o]:
                                        nic_r = r // 2
                                        nic_s = src // 2
                                        if self._is_relay_link(src,r) and len(recv_right_after[nic_r]) and len(send_right_after[nic_s]) and (recv_right_after[nic_r][(c,src,r)] != recv_right_after[nic_r][(o,src,r)] or send_right_after[nic_s][(c,r,src)] != send_right_after[nic_s][(o,r,src)]):
                                            self.is_together_set_0.add((c1,o1,r))
                                            self.is_before_set_1.add((c,o,r))
                                        elif _should_fix_together(src,r,c,o):
                                            self.is_together_set_1.add((c1,o1,r))
                                        elif _should_try_together(src,r,c,o):
                                            print(f'will-try {c} {o} ({src}->{r})')
                                            is_before_ocr = 0
                                            if not extra_heuristic:
                                                if (o,c,r) not in self.is_before:
                                                    self.is_before[(o,c,r)] = opt.addVar(vtype=GRB.BINARY)
                                                is_before_ocr = self.is_before[(o,c,r)]
                                            else:
                                                assert (o,c,r) not in self.is_before
                                            if (c,o,r) not in self.is_before:
                                                self.is_before[(c,o,r)] = opt.addVar(vtype=GRB.BINARY)
                                            if (c1,o1,r) not in self.is_together:
                                                self.is_together[(c1,o1,r)] = opt.addVar(vtype=GRB.BINARY)
                                            opt.addLConstr(self.is_before[(c,o,r)] + self.is_together[(c1,o1,r)] + is_before_ocr == 1)
                                            if j-1>i:
                                                # print((c2,prev_o2,r) in self.is_together, c2,prev_o2, r)
                                                # print((c1,o1,r) in self.is_together)
                                                opt.addLConstr(self.is_together[(c1,o1,r)] <= self.is_together[(c2,prev_o2,r)])
                                        else:
                                            self.is_together_set_0.add((c1,o1,r))
                                            self.is_before_set_1.add((c,o,r))
                                    j = j + 1
                                i = i + 1

        def _add_chunk_order(opt, heuristic, recv_right_after, send_right_after):
            if chunk_order is not None:
                assert(chunk_time is not None)
                assert(len(chunk_order) == R)
                assert(len(chunk_order[0]) == R)
                for r in range(R):
                    for src in self.topology.sources(r):
                        lchunks = [[] for l in range(self.topology.link(src,r))]
                        numl = self.topology.link(src,r)
                        if heuristic == 3 or heuristic == 5:
                            i=0
                            for c in chunk_order[r][src][0]:
                                lchunks[i].append(c)
                                i = (i+1)%numl
                        else:
                            for l in range(numl):
                                lchunks[l] = chunk_order[r][src][l]
                        for l in range(numl):
                            i = 0
                            while i < len(lchunks[l]):
                                j = i + 1
                                c = lchunks[l][i]
                                while j<len(lchunks[l]):
                                    o = lchunks[l][j]
                                    c1,o1 = minmax(c,o)
                                    prev_o = lchunks[l][j-1]
                                    c2, prev_o2 = minmax(c,prev_o)
                                    if r in self.spsets[c] and r in self.spsets[o]:
                                        # Try to send together only if both chunks are sent uninterrupted according to ordering heuristic
                                        if self._is_relay_link(src,r) and len(recv_right_after[r]) and (recv_right_after[r][(c,src)] != recv_right_after[r][(o,src)] or send_right_after[src][(c,r)] != send_right_after[src][(o,r)]):
                                            self.is_together_set_0.add((c1,o1,r))
                                            self.is_before_set_1.add((c,o,r))
                                        elif _should_fix_together(src,r,c,o):
                                            self.is_together_set_1.add((c1,o1,r))
                                        elif _should_try_together(src,r,c,o):
                                            print(f'will-try {c} {o} ({src}->{r})')
                                            is_before_ocr = 0
                                            if not extra_heuristic:
                                                if (o,c,r) not in self.is_before:
                                                    self.is_before[(o,c,r)] = opt.addVar(vtype=GRB.BINARY)
                                                is_before_ocr = self.is_before[(o,c,r)]
                                            else:
                                                assert (o,c,r) not in self.is_before
                                            if (c,o,r) not in self.is_before:
                                                self.is_before[(c,o,r)] = opt.addVar(vtype=GRB.BINARY)
                                            if (c1,o1,r) not in self.is_together:
                                                self.is_together[(c1,o1,r)] = opt.addVar(vtype=GRB.BINARY)
                                            opt.addLConstr(self.is_before[(c,o,r)] + self.is_together[(c1,o1,r)] + is_before_ocr == 1)
                                            if j-1>i:
                                                # Send together with a chunk only if the previous chunks have been sent together as well
                                                opt.addLConstr(self.is_together[(c1,o1,r)] <= self.is_together[(c2,prev_o2,r)])
                                        else:
                                            self.is_together_set_0.add((c1,o1,r))
                                            self.is_before_set_1.add((c,o,r))
                                    j = j + 1
                                i = i + 1

        def alpha(r,dst):
            assert r != dst
            if self._is_relay_link(r,dst):
                alpha = self.topology.remote_alpha
                assert alpha is not None
                return alpha
            return 0

        def beta(r,dst):
            assert r != dst
            if self._is_relay_link(r,dst):
                beta = self.topology.remote_beta
                assert beta is not None
                return beta
            return self.topology.get_invbw(r,dst)

        def calc_latency(src,r,l,c):
            if self._is_relay_link(src,r):
                num_s = 0
                for o in range(C):
                    o1,c1 = minmax(o,c)
                    if (o1,c1,r) in self.is_together_set_1:
                        assert (o1,c1,r) not in self.is_together
                        num_s = num_s + 1
                        continue
                    if (o1,c1,r) in self.is_together_set_0:
                        assert (o1,c1,r) not in self.is_together
                    else:
                        if (o1,c1,r) not in self.is_together:
                            self.is_together[(o1,c1,r)] = opt.addVar(vtype=GRB.BINARY)
                lat = alpha(src,r) + beta(src,r)*(num_s + quicksum(self.is_together[(o,c,r)] if (o,c,r) in self.is_together else 0 for o in range(c)) + quicksum(self.is_together[(c,o,r)] if (c,o,r) in self.is_together else 0 for o in range(c,C)))
                return lat
            return alpha(src,r) + beta(src,r)

        #  Add chunk sent constraints obtained from relaxed encoding
        _add_chunk_sent(opt, heuristic)

        # Weed out cases where is_together is not possible
        for c in self.collective.chunks():
            for r in self.collective.ranks():
                sent_anytime = sum([sum([1 if (c,src,r,l) in self.is_sent_set_1 else 0 for l in range(L)]) for src in self.topology.sources(r)])
                sent_IB = sum([sum([1 if (c,src,r,l) in self.is_sent_set_1 and self._is_relay_link(src,r) else 0 for l in range(L)]) for src in self.topology.sources(r)])
                if sent_anytime == 0:
                    assert (c,c,r) not in self.is_together_set_1
                    assert (c,c,r) not in self.is_together
                    self.is_together_set_0.add((c,c,r))
                else:
                    assert sent_anytime == 1
                    assert (c,c,r) not in self.is_together_set_1
                    assert (c,c,r) not in self.is_together
                    self.is_together_set_1.add((c,c,r))

                if r not in self.spsets[c]:
                    for o in range(C):
                        if o == c:
                            continue
                        o1,c1 = minmax(o,c)
                        if sent_IB:
                            self.is_together_set_0.add((o1,c1,r))
                    continue
                if self.collective.precondition(r, c):
                    for o in range(C):
                        if o == c:
                            continue
                        o1,c1 = minmax(o,c)
                        if sent_IB:
                            self.is_together_set_0.add((o1,c1,r))


        # Add chunk ordering obtained from relaxed encoding
        should_add_switch_order = True
        recv_right_after = {}
        send_right_after = {}
        if should_add_switch_order:
            recv_right_after, send_right_after = _add_switch_order(opt)

        print("RRA", recv_right_after)
        print("SRA", send_right_after)
        if "DGX2" in self.topology.name:
            nic_recv_right_after, nic_send_right_after = _add_nic_order(opt)
            _add_chunk_order_dgx2(opt, heuristic, nic_recv_right_after, nic_send_right_after)
            print("NRRA", nic_recv_right_after)
            print("NSRA", nic_send_right_after)
        else:
            _add_chunk_order(opt, heuristic, recv_right_after, send_right_after)

        def _get_isbefore(c,o,r):
            if (c,o,r) in self.is_before_set_1:
                return True, 1
            elif (c,o,r) in self.is_before:
                return False, self.is_before[(c,o,r)]
            else:
                return True, 0

        def _get_istogether(c,o,r):
            c1,o1 = minmax(c,o)
            if (c1,o1,r) in self.is_together_set_1:
                return True, 1
            elif (c1,o1,r) in self.is_together:
                return False, self.is_together[(c1,o1,r)]
            else:
                return True, 0

        for r in self.collective.ranks():
            src_r = [src for src in self.topology.sources(r)]
            links_r = {src: self.topology.link(src,r) for src in src_r}
            for c in self.collective.chunks():
                opt.addLConstr(self.start[c,r] <= ST)
                if r not in self.spsets[c]:
                    opt.addLConstr(self.start[c,r] == ST)
                    for src in src_r:
                        for l in range(L):
                            opt.addLConstr(self.send[c,src,r,l] == SND)
                    continue
                if self.collective.precondition(r, c):
                    opt.addLConstr(self.start[c,r] == 0)
                else:            
                    for src in src_r:
                        for l in range(links_r[src]):
                            if (c,src,r,l) in self.is_sent_set_1:
                                opt.addLConstr(self.start[c,r] == self.send[c,src,r,l] + calc_latency(src,r,l,c))
                            else:
                                opt.addLConstr(self.send[c,src,r,l] >= SND)
                        for l in range(links_r[src], L):
                            opt.addLConstr(self.send[c,src,r,l] == SND)
                    sent_anytime = sum([sum([1 if (c,src,r,l) in self.is_sent_set_1 else 0 for l in range(links_r[src])]) for src in src_r])
                    if self.collective.postcondition(r, c):
                        opt.addLConstr(self.start[c,r] <= self.time)
                        assert sent_anytime == 1, f'{c} {r} {self.is_sent_set_1}'
                    else:
                        assert sent_anytime <= 1
                        if sent_anytime == 0:
                            opt.addLConstr(self.start[c,r] >= self.time + 1)
                        else:
                            opt.addLConstr(self.start[c,r] <= self.time)

                for src in src_r:
                    for l in range(links_r[src]):
                        if (c,src,r,l) in self.is_sent_set_1:
                            opt.addLConstr(self.start[c,src] <= self.start[c,r])
                        opt.addLConstr(self.start[c,src] <= self.send[c,src,r,l])


                # Order sends from same gpu to same gpu
                for o in range(c):
                    is_static_cor, is_before_cor = _get_isbefore(c,o,r)
                    is_static_ocr, is_before_ocr = _get_isbefore(o,c,r)
                    is_static_t_ocr, is_together_ocr = _get_istogether(o,c,r)
                    if is_static_t_ocr and is_together_ocr == 1:
                        for src in src_r:
                            for l in range(self.topology.link(src,r)):
                                if (c,src,r,l) in self.is_sent_set_1 and (o,src,r,l) in self.is_sent_set_1:
                                    opt.addLConstr(self.send[c,src,r,l] == self.send[o,src,r,l])
                        opt.addLConstr(self.start[c,r] == self.start[o,r])
                    elif not is_static_t_ocr:
                        for src in src_r:
                            for l in range(self.topology.link(src,r)):
                                if (c,src,r,l) in self.is_sent_set_1 and (o,src,r,l) in self.is_sent_set_1:
                                    # if together, same send
                                    opt.addGenConstrIndicator(self.is_together[(o,c,r)], True, self.send[c,src,r,l] == self.send[o,src,r,l])
                        # if together, same start
                        opt.addGenConstrIndicator(self.is_together[(o,c,r)], True, self.start[c,r] == self.start[o,r])

                    if is_static_cor and is_static_ocr and is_static_t_ocr:
                        sent_same = any([1 if (c,src,r,l) in self.is_sent_set_1 and (o,src,r,l) in self.is_sent_set_1 else 0 for l in range(L) for src in self.topology.sources(r)])
                        sent_val = 1 if sent_same else 0
                        assert is_before_cor + is_before_ocr + is_together_ocr == sent_val

                    for src in src_r:
                        for l in range(self.topology.link(src,r)):
                            if (c,src,r,l) in self.is_sent_set_1 and (o,src,r,l) in self.is_sent_set_1:
                                lat_o = calc_latency(src,r,l,o)
                                lat_c = calc_latency(src,r,l,c)

                                if (c,o,r) in self.is_before_set_1:
                                    opt.addLConstr(self.start[c,r] + lat_o <= self.start[o,r])
                                elif (c,o,r) in self.is_before:
                                    opt.addLConstr(self.start[c,r] + lat_o <= self.start[o,r] + M*(1-self.is_before[(c,o,r)]))
                                if (o,c,r) in self.is_before_set_1:
                                    opt.addLConstr(self.start[o,r] + lat_c <= self.start[c,r])
                                elif (o,c,r) in self.is_before:
                                    opt.addLConstr(self.start[o,r] + lat_c <= self.start[c,r] + M*(1-self.is_before[(o,c,r)]))

                sw_added = []
                for src in src_r:
                    if (src,r) in self.topology.switches_involved:
                        l = 0 # TODO
                        for swt_i in self.topology.switches_involved[(src,r)]:
                            if (c,src,r,l) in self.is_sent_set_1:
                                if l not in sw_added:
                                    lat_c = calc_latency(src,r,l,c)
                                    srcs_check = []
                                    for srcs, dsts, _, _, switch_name in self.topology.switches[l]:
                                        if r in dsts and "in" in switch_name:
                                            srcs_check = srcs
                                            break
                                    assert len(srcs_check)>0, f'{r} {c} {src} {l} {self.topology.switches[l]}'
                                    for o in range(c):
                                        for src_o in srcs_check:
                                            if src_o == src or swt_i not in self.topology.switches_involved[(src_o,r)]:
                                                continue
                                            if (o,src_o,r,l) in self.is_sent_set_1:
                                                if o == c:
                                                    assert False
                                                lat_o = calc_latency(src_o,r,l,o)
                                                if (o,c,r,src_o,src) in self.recv_first_set_1:
                                                    opt.addLConstr(self.start[o,r] + lat_c <= self.start[c,r])
                                                elif (c,o,r,src,src_o) in self.recv_first_set_1:
                                                    opt.addLConstr(self.start[c,r] + lat_o <= self.start[o,r])
                                                else:
                                                    assert False, f"no-ordering {o}, {c}, {r}, {src}, {src_o}"
                                                    assert (o,c,r) not in self.recv_first, f'{o},{c},{r}'
                                                    self.recv_first[(o,c,r)] = opt.addVar(vtype=GRB.BINARY)
                                                    opt.addLConstr(self.start[o,r] + lat_c <= self.start[c,r] + M*(1-self.recv_first[(o,c,r)]))
                                                    opt.addLConstr(self.start[c,r] + lat_o <= self.start[o,r] + M*(self.recv_first[(o,c,r)]))
                                    sw_added.append(l)

        for r in self.collective.ranks():
            for c in self.collective.chunks():
                sw_added = []
                for dst in self.topology.destinations(r):
                    if (r,dst) in self.topology.switches_involved:
                        for swt_i in self.topology.switches_involved[(r,dst)]:
                        # for l in range(L):
                            l=0
                            if (c,r,dst,l) in self.is_sent_set_1:
                                if l not in sw_added:
                                    lat_c = calc_latency(r,dst,l,c)
                                    dsts_check = []
                                    for srcs, dsts, _, _, switch_name in self.topology.switches[l]:
                                        if r in srcs and "out" in switch_name:
                                            dsts_check = dsts
                                            break
                                    assert len(dsts_check)>0, f'{r} {c} {dst} {l} {self.topology.switches[l]}'
                                    for o in range(c):
                                        for dst_o in dsts_check:
                                            if dst_o == dst or swt_i not in self.topology.switches_involved[(r,dst_o)]:
                                                continue
                                            if (o,r,dst_o,l) in self.is_sent_set_1:
                                                if o == c:
                                                    assert False
                                                lat_o = calc_latency(r,dst_o,l,o)
                                                if (o,c,r,dst_o,dst) in self.send_first_set_1:
                                                    opt.addLConstr(self.send[o,r,dst_o,l] + lat_o <= self.send[c,r,dst,l])
                                                elif (c,o,r,dst,dst_o) in self.send_first_set_1:
                                                    opt.addLConstr(self.send[c,r,dst,l] + lat_c <= self.send[o,r,dst_o,l])
                                                else:
                                                    assert False
                                                    assert (o,c,r) not in self.send_first, f'{o},{c},{r}'
                                                    self.send_first[(o,c,r)] = opt.addVar(vtype=GRB.BINARY)
                                                    opt.addLConstr(self.send[o,r,dst_o,l] + lat_o <= self.send[c,r,dst,l] + M*(1-self.send_first[(o,c,r)]))
                                                    opt.addLConstr(self.send[c,r,dst,l] + lat_c <= self.send[o,r,dst_o,l] + M*(self.send_first[(o,c,r)]))
                                    sw_added.append(l)


        if "DGX2" in self.topology.name:
            num_local_nodes = R // self.topology.copies
            nic_groups = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15),(16,17),(18,19),(20,21),(22,23),(24,25),(26,27),(28,29),(30,31)]
            # recv on same nic from two different nics in different machine from the first
            for (c,srci,ri,l) in self.is_sent_set_1:
                assert l == 0
                nic_recv = nic_groups[ri // 2]
                nic_send = nic_groups[srci // 2]
                if nic_recv[0]//num_local_nodes != nic_send[0]//num_local_nodes:
                    lat_c = calc_latency(srci,ri,0,c)
                    for o in range(c):
                        for rj in nic_recv:
                            if rj == ri:
                                continue
                            for srcj in range(R):
                                if srcj//num_local_nodes != ri//num_local_nodes and srcj not in nic_send and (o,srcj,rj,0) in self.is_sent_set_1:
                                    lat_o = calc_latency(srcj,rj,0,o)
                                    if (o,c,rj,srcj,ri,srci) in self.nic_recv_first_set_1:
                                        opt.addLConstr(self.start[o,rj] + lat_c <= self.start[c,ri])
                                    elif (c,o,ri,srci,rj,srcj) in self.nic_recv_first_set_1:
                                        opt.addLConstr(self.start[c,ri] + lat_o <= self.start[o,rj])
                                    else:
                                        assert False

            # send from on same nic to two different nics in different machine from the first
            for (c,ri,dsti,l) in self.is_sent_set_1:
                assert l == 0
                nic_recv = nic_groups[dsti // 2]
                nic_send = nic_groups[ri // 2]
                if nic_recv[0]//num_local_nodes != nic_send[0]//num_local_nodes:
                    lat_c = calc_latency(ri,dsti,0,c)
                    for o in range(c):
                        for rj in nic_send:
                            if rj == ri:
                                continue
                            for dstj in range(R):
                                if dstj//num_local_nodes != ri//num_local_nodes and dstj not in nic_recv and (o,rj,dstj,0) in self.is_sent_set_1:
                                    lat_o = calc_latency(rj,dstj,0,o)
                                    if (o,c,rj,dstj,ri,dsti) in self.nic_send_first_set_1:
                                        opt.addLConstr(self.send[o,rj,dstj,0] + lat_o <= self.send[c,ri,dsti,0])
                                    elif (c,o,ri,dsti,rj,dstj) in self.nic_send_first_set_1:
                                        opt.addLConstr(self.send[c,ri,dsti,0] + lat_c <= self.send[o,rj,dstj,0])
                                    else:
                                        assert False


    def optimize(self, num_chunks_per_node=1, chunk_order=None, chunk_time=None, switch_chunk_order_recv=None, switch_chunk_time_recv=None, switch_chunk_order_send=None, switch_chunk_time_send=None,  nic_chunk_order_recv=None, nic_chunk_time_recv=None, nic_chunk_order_send=None, nic_chunk_time_send=None, heuristic=4):
        import math
        from time import time
        print(self.topology.name)
        self.collective = self.collective_og.chunk_up(num_chunks_per_node)
        start_time = time()
        opt = Model('sccl_{}_{}'.format(self.topology.name, self.collective.name))
        self._encode(opt, chunk_order, chunk_time, 
            switch_chunk_order_recv, switch_chunk_time_recv, switch_chunk_order_send, switch_chunk_time_send,
            nic_chunk_order_recv, nic_chunk_time_recv, nic_chunk_order_send, nic_chunk_time_send, heuristic)
        # print('Encoded', flush=True)
        opt.optimize()
        end_time = time()
        print("strict time (encode+solve)", end_time-start_time, flush=True)

        if opt.status == GRB.INFEASIBLE:
            opt.computeIIS()
            opt.write("model.ilp")
            raise ValueError("Infeasible model")

        C = self.collective.num_chunks
        R = self.collective.num_nodes
        L = 0
        for src in range(R):
            for dst in self.topology.destinations(src):
                if self.topology.link(src,dst) > L:
                    L = self.topology.link(src,dst)

        send_dict = defaultdict(list)
        SCALE_TIME = 10 # Used because alpha/beta costs may be floats, should scale and then convert to int

        model_str = ""
        for c in range(C):
            for r in range(R):
                if self.start[c,r].X <= self.time.X + 0.005:
                    model_str += f'start[{c},{r}]={self.start[c,r].X}\n'
        for r in range(R):
            for src in self.topology.sources(r):
                for l in range(L):
                    for c_np in chunk_order[r][src][l]:
                        c = int(c_np)
                        assert (c,src,r,l) in self.is_sent_set_1
                        model_str += f'{c}: {src} --{l}--> {r}  t={self.send[c,src,r,l].X}\n'
                        t = int(SCALE_TIME*self.send[c,src,r,l].X + 0.0001)
                        send_dict[t].append([c,src,r,t,l])
                    for c_np in range(C):
                        c = int(c_np)
                        if c not in chunk_order[r][src][l]:
                            assert (c,src,r,l) not in self.is_sent_set_1
        for c in range(C):
            for o in range(c):
                for r in range(R):
                    if (o,c,r) in self.is_together:
                        if self.is_together[(o,c,r)].X >= 0.995:
                            model_str += f'({c},{o},{r})\n'
                    elif (o,c,r) in self.is_together_set_1:
                        model_str += f'({c},{o},{r}) set\n'

        print(model_str)
        steps=[]
        send_times = sorted(send_dict.keys())
        i = 0
        while(i < len(send_times)):
            num_sends = [[0 for _ in range(R)] for _ in range(R)]
            j = i + 1
            while j < len(send_times):
                to_break = False
                t_end = send_times[j]
                for (c,src,r,_,_) in send_dict[t_end]:
                    for t in range(i,j):
                        for (ci,_,ri,_,_) in send_dict[send_times[t]]:
                            if c == ci and src == ri:
                                to_break = True
                                break
                        if to_break:
                            break
                    if to_break:
                        break
                if to_break:
                    break
                j = j + 1
            sends = []
            for k in range(i,j):
                sends.extend(send_dict[send_times[k]])
            num_sends = [[0 for _ in range(R)] for _ in range(R)]
            for (c,src,r,_,_) in sends:
                num_sends[r][src] = num_sends[r][src] + 1
            rounds = 0
            for srcs, dsts, bw, name in self.topology.real_bandwidth_constraints():
                util = 0
                for dst in dsts:
                    for src in srcs:
                        util += num_sends[dst][src]
                if rounds <= util * bw * SCALE_TIME:
                    rounds = math.ceil(util * bw * SCALE_TIME)
            steps.append(Step(rounds, sorted(sends, key=lambda x: x[3])))
            i = j

        instance = Instance(
            steps=len(steps),
            extra_rounds=0,
            chunks=num_chunks_per_node,
        )
        soltype = "e" if chunk_order is None else "improve"
        from time import time
        timestamp = int(time())
        np.save(f'send_dict_{timestamp}.npy', send_dict)
        return Algorithm.make_implementation(self.collective_og, self.topology, instance, steps, f'-gurobisol-{soltype}-{timestamp}')


    def check_sol(self, num_chunks_per_node=1,ts=""):
        import math
        self.collective = self.collective_og.chunk_up(num_chunks_per_node)

        C = self.collective.num_chunks
        R = self.collective.num_nodes
        L = 0
        for src in range(R):
            for dst in self.topology.destinations(src):
                if self.topology.link(src,dst) > L:
                    L = self.topology.link(src,dst)
        print(R,C,L)
        SCALE_TIME = 10 # Used because alpha/beta costs may be floats, should scale and then convert to int

        if len(ts):
            send_dict = np.load(f"send_dict_{ts}.npy", allow_pickle=True).item()
        else:
            send_dict = np.load("send_dict.npy", allow_pickle=True).item()
        steps=[]
        send_times = sorted(send_dict.keys())
        i = 0
        while(i < len(send_times)):
            do_print = False
            t = send_times[i]
            for (c,src,r,_,_) in send_dict[t]:
                if self._is_relay_link(src,r):
                    do_print = True
                    print(f'({t},{src},{r},{c})', end=" ")
            if do_print:
                print("")
            i = i+1
        i = 0
        while(i < len(send_times)):
            j = i + 1
            while j < len(send_times):
                to_break = False
                t_end = send_times[j]
                for (c,src,r,_,_) in send_dict[t_end]:
                    for t in range(i,j):
                        for (ci,_,ri,_,_) in send_dict[send_times[t]]:
                            if c == ci and src == ri:
                                to_break = True
                                break
                        if to_break:
                            break
                    if to_break:
                        break
                if to_break:
                    break
                j = j + 1
            sends = []
            for k in range(i,j):
                sends.extend(send_dict[send_times[k]])
            num_sends = [[0 for _ in range(R)] for _ in range(R)]
            for (c,src,r,_,_) in sends:
                num_sends[r][src] = num_sends[r][src] + 1
            rounds = 0
            for srcs, dsts, bw, name in self.topology.real_bandwidth_constraints():
                util = 0
                for dst in dsts:
                    for src in srcs:
                        util += num_sends[dst][src]
                if rounds <= util * bw * SCALE_TIME:
                    rounds = math.ceil(util * bw * SCALE_TIME)
            steps.append(Step(rounds, sorted(sends, key=lambda x: x[3] )))
            i = j

        instance = Instance(
            steps=len(steps),
            extra_rounds=0,
            chunks=num_chunks_per_node,
        )
        soltype = "echecksol"
        from time import time
        timestamp = int(time())
        return Algorithm.make_implementation(self.collective_og, self.topology, instance, steps, f'-gurobisol-{soltype}-{timestamp}')


    def optimize_double(self, chunk):
        opt = [None for i in range(7)]
        for num_chunks_per_node in [3]:
            i = num_chunks_per_node
            self.collective = self.collective_og.chunk_up(num_chunks_per_node)
            opt[i] = Model('sccl_{}_{}'.format(self.topology.name, self.collective.name))
            self._encode(opt[i], i)
            print('Encoded', flush=True)
            opt[i].optimize()

            if opt[i].status == GRB.INFEASIBLE:
                opt[i].computeIIS()
                opt[i].write("model.ilp")
                raise ValueError("Infeasible model")

            C = self.collective.num_chunks
            R = self.collective.num_nodes
            L = 0
            for src in range(R):
                for dst in self.topology.destinations(src):
                    if self.topology.link(src,dst) > L:
                        L = self.topology.link(src,dst)

            self.collective = self.collective_og.chunk_up(2*num_chunks_per_node)
            opt[2*i] = Model('sccl_{}_{}'.format(self.topology.name, self.collective.name))
            self._encode(opt[2*i], 2*i)

            add_factor = lat_mult_factor / (2*i)

            model_str = ""
            # new_model_str
            for c in range(C):
                for r in range(R):
                    if self.start[c,r].X <= 48:
                        model_str += f'start[{c},{r}]={self.start[c,r].X}\n'
                        # for c_new in range(2*c, 2*(c+1)):
                        if self.start[c,r].X < 0.001:
                            self.start[2*i][2*c,r].start = 0
                            self.start[2*i][2*c+1,r].start = 0
                            model_str += f'=> start[{2*c},{r}] = 0\n'
                            model_str += f'=> start[{2*c+1},{r}] = 0\n'
                        else:
                            self.start[2*i][2*c,r].start = int(self.start[c,r].X+0.05) - add_factor
                            self.start[2*i][2*c+1,r].start =  int(self.start[c,r].X+0.05)
                            model_str += f'=> start[{2*c},{r}] = {int(self.start[c,r].X+0.05)-add_factor}\n'
                            model_str += f'=> start[{2*c+1},{r}] = {int(self.start[c,r].X+0.05)}\n'
            for c in range(C):
                for r in range(R):
                    for src in self.topology.sources(r):
                        for l in range(L):
                            if self.is_sent[c,src,r,l].X >= 0.995:
                                model_str += f'{c}: {src} --{l}--> {r}  t={self.send[c,src,r,l].X}\n'
                                self.is_sent[2*i][2*c,src,r,l].start = 1
                                self.is_sent[2*i][2*c+1,src,r,l].start = 1
                                # if self.send[c,src,r,l].X > 0.001:
                                self.send[2*i][2*c,src,r,l].start = int(self.send[c,src,r,l].X + 0.05)
                                self.send[2*i][2*c+1,src,r,l].start = int(self.send[c,src,r,l].X + 0.05) + add_factor
                                model_str += f'=> {2*c} : {src} --{l}--> {r} t={int(self.send[c,src,r,l].X + 0.05)}\n'
                                model_str += f'=> {2*c+1} : {src} --{l}--> {r} t={int(self.send[c,src,r,l].X + 0.05)+add_factor}\n'

            print(model_str)

            opt[2*i].optimize()

            model_str = ""
            C = self.collective.num_chunks
            for c in range(C):
                for r in range(R):
                    if self.start[2*i][c,r].X <= 500:
                        model_str += f'start[{c},{r}]={self.start[2*i][c,r].X}\n'

            for c in range(C):
                for r in range(R):
                    for src in self.topology.sources(r):
                        for l in range(L):
                            if self.is_sent[2*i][c,src,r,l].X >= 0.995:
                                model_str += f'{c}: {src} --{l}--> {r}  t={self.send[2*i][c,src,r,l].X}\n'
        print(model_str)
        return model_str