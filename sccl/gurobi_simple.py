# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import defaultdict
from gurobipy import GRB, Model, quicksum, abs_, and_
import numpy as np
import os
import pickle as pkl
from sccl.algorithm import *
from sccl.shortest_path_sets import *
from sccl.topologies.topology import DistributedTopology, MachineTopology
from time import time


class GurobiOptimizerSimple(object):
    def __init__(self, topology, collective):
        self.topology = topology
        self.collective_og = collective
        self.send = [None, None]
        self.start = [None, None]
        self.time = [None, None]
        self.is_sent = [None, None]

    def _is_relay_link(self,r,dst):
        if isinstance(self.topology, DistributedTopology) and self.topology.m_top == MachineTopology.RELAYED:
            copies = self.topology.copies
            num_local_nodes = self.topology.num_nodes() // copies
            if r % num_local_nodes in self.topology.relay[0] and dst % num_local_nodes in self.topology.relay[1] and r // num_local_nodes != dst // num_local_nodes:
                return True

    def latency(self, src, dst, l):
        return self.topology.get_invbw(src,dst)

    def bw(self, src, dst, l):
        return self.topology.get_invbw(src,dst)

    # [Deprecated] Ensure is_before constraints hold true in the solution
    def _sanity_check(self, time_recv, chunk_recv, time_send, chunk_send, time_algo):
        np.set_printoptions(precision=10)
        violations = defaultdict(list)
        for r in range(self.collective.num_nodes):
            for src in self.topology.sources(r):
                for l in range(self.topology.link(src,r)):
                    lat = self.latency(r,src,l)
                    # Chunks are received one after the other (with some tolerance)
                    if len(time_recv[r][src][l]) > 1:
                        try:
                            assert (time_recv[r][src][l][1:] - time_recv[r][src][l][:-1] >= lat-0.0001).all() , (r, src, l, chunk_recv[r][src][l], time_recv[r][src][l], lat)
                            assert (time_recv[r][src][l][1:] - time_recv[r][src][l][:-1] <= lat+0.0001).all() , (r, src, l, chunk_recv[r][src][l], time_recv[r][src][l], lat)
                        except Exception as e:
                            recv_bw = (time_recv[r][src][l][1:] - time_recv[r][src][l][:-1] >= lat-0.0001) & (time_recv[r][src][l][1:] - time_recv[r][src][l][:-1] <= lat+0.0001)
                            for i, val in enumerate(recv_bw):
                                if not val:
                                    violations["recv_bw"].append([f'{src}->{r}',l,chunk_recv[r][src][l][i], f't_r1={time_recv[r][src][l][i]}', chunk_recv[r][src][l][i+1], f't_r2={time_recv[r][src][l][i+1]}'])

                    # Chunks are received in the same order they are sent
                    try:
                        assert (chunk_recv[r][src][l] == chunk_send[src][r][l]).all() , (r, src, l, chunk_recv[r][src][l], chunk_send[src][r][l])
                    except Exception as e:
                        chunk_transfer = (chunk_recv[r][src][l] == chunk_send[src][r][l])
                        for i, val in enumerate(chunk_transfer):
                            if not val:
                                violations["chunk_transfer"].append([f'{src}->{r}', l, chunk_send[src][r][l][i], f't_s={time_send[src][r][l][i]}', chunk_recv[r][src][l][i], f't_r={time_recv[r][src][l][i]}'])
                    # Chunks sent are received in time (with some tolerance)
                    try:
                        assert (time_recv[r][src][l] - time_send[src][r][l] <= lat + 0.0001).all() , (time_recv[r][src][l] - time_send[src][r][l], lat, (r,src,l,chunk_send[r][src][l]))
                        assert (time_recv[r][src][l] - time_send[src][r][l] >= lat - 0.0001).all() , (time_recv[r][src][l], time_send[src][r][l], lat, (r,src,l,chunk_send[r][src][l]))
                    except Exception as e:
                        time_transfer = (time_recv[r][src][l] - time_send[src][r][l] <= lat + 0.0001) & (time_recv[r][src][l] - time_send[src][r][l] >= lat - 0.0001)
                        for i, val in enumerate(time_transfer):
                            if not val:
                                violations["time_transfer"].append([f'{src}->{r}', l, chunk_send[src][r][l][i], f't_s={time_send[src][r][l][i]}', chunk_recv[r][src][l][i], f't_r={time_recv[r][src][l][i]}'])

            for dst in self.topology.destinations(r):
                for l in range(self.topology.link(r,dst)):
                    lat = self.latency(r,dst,l)
                    # Chunks are sent one after the other
                    if len(time_send[r][dst][l]) > 1:
                        try:
                            assert (time_send[r][dst][l][1:] - time_send[r][dst][l][:-1] >= lat-0.0001).all() , (r,dst,time_send[r][dst][l], lat)
                        except Exception as e:
                            send_bw = time_send[r][dst][l][1:] - time_send[r][dst][l][:-1] >= lat-0.0001
                            for i, val in enumerate(send_bw):
                                if not val:
                                    violations["send_bw"].append([f'{r}->{dst}',l,chunk_send[r][dst][l][i], f't_s1={time_send[r][dst][l][i]}', chunk_send[r][dst][l][i+1], f't_s2={time_send[r][dst][l][i+1]}'])

        for step in range(int(time_algo+0.0001)):
            for r in range(self.collective.num_nodes):
                for dst in self.topology.destinations(r):
                    for l in range(self.topology.link(dst,r)):
                        for i, t in enumerate(time_send[r][dst][l]):
                            if (t < step + 0.0001) and (t > step - 0.0001):
                                print(f'{r} -> {dst} l={l}, c={chunk_send[r][dst][l][i]}, t={time_send[r][dst][l][i]}')

        print("violations:")
        print(violations)
        return True

    def _encode(self, opt, iid, heuristic=3):
        C = self.collective.num_chunks
        R = self.collective.num_nodes
        M = 10000000 # big-M for maximum self.time[iid] between sends
        ST = 500000 # self.time[iid] for unsent sends and unstarted starts
        SND = 1000000 # self.time[iid] for unsent sends and unstarted starts
        opt.Params.Method = 2
        opt.Params.Threads = 1
        opt.Params.MIPFocus = 1 # Gives good feasible solutions quickly, slow in generating proof

        L = 0
        for src in range(R):
            for dst in self.topology.destinations(src):
                if self.topology.link(src,dst) > L:
                    L = self.topology.link(src,dst)

        self.send[iid] = opt.addVars(C, R, R, L, name="send", vtype=GRB.CONTINUOUS, lb=0.0)
        self.start[iid] = opt.addVars(C, R, name="start", vtype=GRB.CONTINUOUS, lb=0.0)
        self.time[iid] = opt.addVar(name="time", vtype=GRB.CONTINUOUS)
        opt.addLConstr(self.time[iid] <= ST-1)
        self.is_sent[iid] = opt.addVars(C,R,R,L, name="is_sent", vtype=GRB.BINARY)

        opt.setObjective(self.time[iid], GRB.MINIMIZE)


        # Not sure what to do for allgather in DGX2
        def _add_relay_relaxation(opt):
            assert isinstance(self.topology, DistributedTopology)
            assert self.topology.m_top == MachineTopology.RELAYED

            copies = self.topology.copies
            num_local_nodes = self.topology.num_nodes() // copies
            num_local_chunks = self.collective.num_chunks // copies

            for c in self.collective.chunks():
                pair_set = defaultdict(set)
                for r1 in self.collective.pre_on(c):
                    for r2 in self.collective.post_on(c):
                        snd_node = r1 // num_local_nodes
                        rcv_node = r2 // num_local_nodes
                        if snd_node != rcv_node:
                            #  TODO make it use the closest relay gpu (diff for ITP and DGX2)
                            if "DGX2" in self.topology.name:
                                pair_set[(snd_node,rcv_node)].add((r1,r2))
                            elif "DGX1" in self.topology.name:
                                snd_gpu = snd_node * num_local_nodes + self.topology.relay[0][0]
                                rcv_gpu = rcv_node * num_local_nodes + self.topology.relay[1][0]
                                pair_set[(snd_node,rcv_node)].add((snd_gpu,rcv_gpu))
                            else:
                                assert False, "Topology not supported"
                for (snode, rnode) in pair_set:
                    if len(pair_set[(snode,rnode)]):
                        opt.addLConstr(quicksum(self.is_sent[iid][c,src,r,0] for (src,r) in pair_set[(snode,rnode)]) >= 1)
                # Set rest of the source to destination IB sends to 0
                for src1 in range(num_local_nodes):
                    for r1 in range(num_local_nodes):
                        for i in range(copies):
                            for j in range(copies):
                                if i == j:
                                    continue
                                src = src1 + i * num_local_nodes
                                r = r1 + j * num_local_nodes
                                if (src,r) not in pair_set[(i,j)]:
                                    opt.addLConstr(self.is_sent[iid][c,src,r,0] == 0)

        # Add symmetric actions to distributed machines
        def _add_symmetry(opt, L):
            assert isinstance(self.topology, DistributedTopology)
            # assert self.topology.m_top == MachineTopology.RELAYED
            copies = self.topology.copies
            num_nodes = self.topology.num_nodes()
            num_chunks = self.collective.num_chunks
            num_local_nodes = self.topology.num_nodes() // copies
            num_local_chunks = self.collective.num_chunks // copies
            chunk_per_node = num_local_chunks // num_local_nodes

            # alltoall symmetry
            if "Alltoall" in self.collective.name:
                # print("Added alltoall symmetry")
                for r1 in range(num_local_nodes):
                    for dst in range(num_nodes):
                        for c_up in range(self.chunkup):
                            # Get the chunk which has precondition on r1 in the local node and postcondition on dst anywhere
                            c = (dst * (chunk_per_node//self.chunkup) + r1)*self.chunkup + c_up
                            assert self.collective.precondition(r1, c) and self.collective.postcondition(dst, c)
                            for r in range(num_nodes):
                                for src in self.topology.sources(r):
                                    for l in range(L):
                                        for i in range(1,copies):
                                            r_copy = (r + i*num_local_nodes) % num_nodes
                                            src_copy = (src + i*num_local_nodes) % num_nodes
                                            c_copy = (((dst + i*num_local_nodes) % num_nodes) * (chunk_per_node//self.chunkup) + ((r1 + i*num_local_nodes) % num_nodes))*self.chunkup + c_up
                                            opt.addLConstr(self.send[iid][c,src,r,l] == self.send[iid][c_copy, src_copy, r_copy, l])
                                            opt.addLConstr(self.is_sent[iid][c,src,r,l] == self.is_sent[iid][c_copy, src_copy, r_copy, l])
                                for i in range(1,copies):
                                    r_copy = (r + i*num_local_nodes) % num_nodes
                                    c_copy = (((dst + i*num_local_nodes) % num_nodes) * (chunk_per_node//self.chunkup) + ((r1 + i*num_local_nodes) % num_nodes))*self.chunkup + c_up
                                    opt.addLConstr(self.start[iid][c,r] == self.start[iid][c_copy, r_copy])
                return

            # allgather symmetry
            if "Allgather" in self.collective.name:
                # print("Added allgather symmetry")
                for c in range(num_local_chunks):
                    for r in range(num_nodes):
                        for src in self.topology.sources(r):
                            for l in range(L):
                                for i in range(1,copies):
                                    r_copy = (r + i*num_local_nodes) % num_nodes
                                    src_copy = (src + i*num_local_nodes) % num_nodes
                                    c_copy = c + i*num_local_chunks
                                    opt.addLConstr(self.send[iid][c,src,r,l] == self.send[iid][c_copy, src_copy, r_copy, l])
                                    opt.addLConstr(self.is_sent[iid][c,src,r,l] == self.is_sent[iid][c_copy, src_copy, r_copy, l])

                        for i in range(1,copies):
                            r_copy = (r + i*num_local_nodes) % num_nodes
                            c_copy = c + i*num_local_chunks
                            opt.addLConstr(self.start[iid][c,r] == self.start[iid][c_copy, r_copy])
                return

            assert False , f'Symmetry conditions not specified for {self.collective.name}'

        for c in self.collective.chunks():
            for r in self.collective.ranks():
                opt.addLConstr(self.start[iid][c,r] <= ST)
                # Fixing to only spsets will reduce chances for contiguity, but it helps scale
                if r not in self.spsets[c]:
                    opt.addLConstr(self.start[iid][c,r] == ST)
                    for src in self.topology.sources(r):
                        for l in range(L):
                            opt.addLConstr(self.send[iid][c,src,r,l] == SND)
                            opt.addLConstr(self.is_sent[iid][c,src,r,l] == 0)
                    continue
                for src in self.topology.sources(r):
                    for l in range(self.topology.link(src,r), L):
                        opt.addLConstr(self.is_sent[iid][c,src,r,l] == 0)

                if self.collective.precondition(r, c):
                    # Have chunks start on their starting ranks before the first step
                    opt.addLConstr(self.start[iid][c,r] == 0)
                    for src in self.topology.sources(r):
                        for l in range(self.topology.link(src,r)):
                            opt.addLConstr(self.is_sent[iid][c,src,r,l] == 0)
                else:
                    for src in self.topology.sources(r):
                        for l in range(self.topology.link(src,r)):
                            opt.addGenConstrIndicator(self.is_sent[iid][c,src,r,l], True, self.start[iid][c,r] == self.send[iid][c,src,r,l] + self.latency(src,r,l))
                            opt.addGenConstrIndicator(self.is_sent[iid][c,src,r,l], False, self.send[iid][c,src,r,l] == SND)
                        for l in range(self.topology.link(src,r), L):
                            opt.addLConstr(self.is_sent[iid][c,src,r,l] == 0)
                            opt.addLConstr(self.send[iid][c,src,r,l] == SND)
                    # self.is_sent[iid][c,src,r,l] = 1 if it is feasible to send chunk c from src to r using link l
                    if self.collective.postcondition(r, c):
                        opt.addLConstr(quicksum(quicksum(self.is_sent[iid][c,src,r,l] for l in range(L)) for src in self.topology.sources(r)) == 1)
                        opt.addLConstr(self.start[iid][c,r] <= self.time[iid])
                    else:
                        opt.addLConstr(quicksum(quicksum(self.is_sent[iid][c,src,r,l] for l in range(L)) for src in self.topology.sources(r)) <= 1)
                        opt.addLConstr(self.start[iid][c,r] <= self.time[iid] + M*(1-quicksum(quicksum(self.is_sent[iid][c,src,r,l] for l in range(L)) for src in self.topology.sources(r))))
                        opt.addLConstr(self.start[iid][c,r] >= self.time[iid] + 1 - M*(quicksum(quicksum(self.is_sent[iid][c,src,r,l] for l in range(L)) for src in self.topology.sources(r))))

                for src in self.topology.sources(r):
                    for l in range(self.topology.link(src,r)):
                        opt.addLConstr(self.start[iid][c,src] <= self.send[iid][c,src,r,l])

        # Hardcode for 2 DGX2s
        # Assumes that nic is common for same sends and recvs
        if "DGX2" in self.topology.name:
            num_local_nodes = R // self.topology.copies
            num_nic = R // 2
            nic_groups = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15),(16,17),(18,19),(20,21),(22,23),(24,25),(26,27),(28,29),(30,31)]

            num_recv_nic = num_nic
            num_send_nic = num_nic
            t_recv = opt.addVars(num_recv_nic, num_send_nic, name="t_recv", vtype=GRB.CONTINUOUS, lb=0.0)
            t_send = opt.addVars(num_send_nic, num_recv_nic, name="t_send", vtype=GRB.CONTINUOUS, lb=0.0)
            for ni, nic_recv in enumerate(nic_groups):
                nj_list = []
                for nj, nic_send in enumerate(nic_groups):
                    if nic_send[0]//num_local_nodes  != nic_recv[0]//num_local_nodes:
                        nj_list.append(nj)
                        for ri in nic_recv:
                            opt.addLConstr(t_recv[ni,nj] >= quicksum(quicksum(self.latency(srcj,ri,0) * self.is_sent[iid][c,srcj,ri,0] for srcj in nic_send) for c in self.collective.chunks()))
                opt.addLConstr(self.time[iid] >= quicksum(t_recv[ni,nj] for nj in nj_list))
            # Lower bound time with max sends and receives in a NIC group
            for ni, nic_send in enumerate(nic_groups):
                nj_list = []
                for nj, nic_recv in enumerate(nic_groups):
                    if nic_send[0]//num_local_nodes  != nic_recv[0]//num_local_nodes:
                        nj_list.append(nj)
                        for ri in nic_send:
                            opt.addLConstr(t_send[ni,nj] >= quicksum(quicksum(self.latency(ri,dstj,0) * self.is_sent[iid][c,ri,dstj,0] for dstj in nic_recv) for c in self.collective.chunks()))
                opt.addLConstr(self.time[iid] >= quicksum(t_send[ni,nj] for nj in nj_list))

        # Lower bound time with max sends and receives in a gpu on a switch
        for switches in self.topology.switches:
            l = 0
            for srcs, dsts, _, _, switch_name in switches:
                if "in" in switch_name:
                    for dst in dsts:
                        if heuristic == 4:
                            opt.addLConstr(self.time[iid] >= quicksum(quicksum(self.bw(srci,dst,l)*self.is_sent[iid][c,srci,dst,l] for c in range(C)) for srci in srcs))
                        else:
                            opt.addLConstr(self.time[iid] >= quicksum(quicksum(self.latency(srci,dst,l)*self.is_sent[iid][c,srci,dst,l] for c in range(C)) for srci in srcs))
                if "out" in switch_name:
                    for src in srcs:
                        if heuristic == 4:
                            opt.addLConstr(self.time[iid] >= quicksum(quicksum(self.bw(src,dsti,l)*self.is_sent[iid][c,src,dsti,l] for c in range(C)) for dsti in dsts))
                        else:
                            opt.addLConstr(self.time[iid] >= quicksum(quicksum(self.latency(src,dsti,l)*self.is_sent[iid][c,src,dsti,l] for c in range(C)) for dsti in dsts))
            l = l + 1

        # Lower bound time with max sends and receives between two gpus
        for r in self.collective.ranks():
            for src in self.topology.sources(r):
                for l in range(self.topology.link(src,r)):
                    if heuristic == 4:
                        opt.addLConstr(self.time[iid] >= quicksum(self.bw(src,r,l)*self.is_sent[iid][c,src,r,l] for c in range(C)))
                    else:
                        opt.addLConstr(self.time[iid] >= quicksum(self.latency(src,r,l)*self.is_sent[iid][c,src,r,l] for c in range(C)))

        if isinstance(self.topology, DistributedTopology):
            _add_symmetry(opt,L)
            if self.topology.m_top == MachineTopology.RELAYED:
                if iid == 0:
                    _add_relay_relaxation(opt)

    # All paths that a chunk follows in the solution
    def set_paths(self, chunk_send):
        paths = defaultdict(list)
        def has_next(c,r):
            for dst in self.topology.destinations(r):
                for l in range(self.topology.link(r,dst)):
                    if c in chunk_send[r][dst][l]:
                        return True
            return False

        def prev(c,r):
            if self.collective.precondition(r, c):
                return -1
            for src in self.topology.sources(r):
                for l in range(self.topology.link(src,r)):
                    if c in chunk_send[src][r][l]:
                        return src
            return -2

        for r in range(self.collective.num_nodes):
            for src in self.topology.sources(r):
                for l in range(self.topology.link(src,r)):
                    for c in chunk_send[src][r][l]:
                        if not has_next(c,r):
                            path = [(src,r,self.bw(src,r,0))]
                            p_r = src
                            while True:
                                p_src = prev(c,p_r)
                                if p_src == -1:
                                    break
                                elif p_src == -2:
                                    assert False
                                else:
                                    path.append((p_src,p_r,self.bw(p_src,p_r,0)))
                                    p_r = p_src
                            paths[c].append(path)
        return paths

    def optimize(self, chunkup=1, heuristic=4):
        self.collective = self.collective_og.chunk_up(chunkup)
        self.spsets = shortest_path_sets(self.topology, self.collective)
        self.chunkup = chunkup
        instance_name = 'sccl_{}_{}_gurobiSimple'.format(self.topology.name, self.collective.name)

        start_time = time()
        opt = Model(instance_name)
        self._encode(opt, 0, heuristic)
        opt.optimize()
        end_time = time()
        print("simple time (encode+solve)", end_time-start_time, flush=True)

        if opt.status == GRB.INFEASIBLE:
            opt.computeIIS()
            opt.write(f'model_{instance_name}.ilp')
            raise ValueError("Infeasible model")

        C = self.collective.num_chunks
        R = self.collective.num_nodes
        L = 0
        for src in range(R):
            for dst in self.topology.destinations(src):
                if self.topology.link(src,dst) > L:
                    L = self.topology.link(src,dst)

        # Get chunk ordering using heuristics to feed into strict solver
        time_recv = [[[[] for l in range(L)] for src in range(R)] for r in range(R)]
        chunk_recv = [[[[] for l in range(L)] for src in range(R)] for r in range(R)]
        time_send = [[[[] for l in range(L)] for src in range(R)] for r in range(R)]
        chunk_send = [[[[] for l in range(L)] for src in range(R)] for r in range(R)]
        if "DGX2" in self.topology.name and isinstance(self.topology, DistributedTopology):
            # Hardcode that DGX2s have external and internal switches
            LL = L + 1
        else:
            LL = L
        switch_time_recv = [[[] for l in range(LL)] for r in range(R)]
        switch_chunk_recv = [[[] for l in range(LL)] for r in range(R)]
        switch_time_send = [[[] for l in range(LL)] for r in range(R)]
        switch_chunk_send = [[[] for l in range(LL)] for r in range(R)]
        nic_time_recv = [[[] for l in range(L)] for r in range(R//2)]
        nic_chunk_recv = [[[] for l in range(L)] for r in range(R//2)]
        nic_time_send = [[[] for l in range(L)] for r in range(R//2)]
        nic_chunk_send = [[[] for l in range(L)] for r in range(R//2)]

        def argsort(s,p):
            indexes = s.argsort()
            return s[indexes], p[indexes]

        model_str = ""
        for c in range(C):
            for r in range(R):
                if self.start[0][c,r].X <= self.time[0].X + 0.005:
                    model_str += f'start[{c},{r}]={self.start[0][c,r].X}\n'
        # incoming_nodes = [{1} for r in range(R)]
        for c in range(C):
            for r in range(R):
                for src in self.topology.sources(r):
                    for l in range(L):
                        if self.is_sent[0][c,src,r,l].X >= 0.995:
                            model_str += f'{c}: {src} --{l}--> {r}  t={self.send[0][c,src,r,l].X}\n'
                            if heuristic == 3:
                                chunk_send[src][r][0].append(c)
                                time_send[src][r][0].append(int(self.send[0][c,src,r,l].X + 0.005))
                                chunk_recv[r][src][0].append(c)
                                time_recv[r][src][0].append(int(self.start[0][c,r].X + 0.005))
                            else:
                                chunk_send[src][r][l].append(c)
                                time_send[src][r][l].append(int(self.send[0][c,src,r,l].X + 0.005))
                                chunk_recv[r][src][l].append(c)
                                time_recv[r][src][l].append(int(self.start[0][c,r].X + 0.005))

        print(model_str)
        paths = self.set_paths(chunk_send)
        for c in paths:
            print("paths",c,paths[c])

        # Populate data for heuristic ordering
        # how much chunk c needs to travel starting from src when travelling on path with (src,dst)
        def to_travel(c,src,dst):
            for path in paths[c]:
                to_travel_bw = 0
                for (srci,dsti,bw) in path:
                    to_travel_bw = to_travel_bw + self.latency(srci,dsti,0)
                    if srci == src and dsti == dst:
                        return to_travel_bw
                        to_t = 0
                        for j in range(i):
                            to_t = to_t + path[j][2]
                        return to_t
            assert False, f'missing {c} in {src} -> {dst}'
        # how much chunk c has travelled utpo src when travelling on path with (src,dst)
        def has_travelled(c,src,dst):
            for path in paths[c]:
                has_travelled_bw = 0
                for (srci,dsti,bw) in reversed(path):
                    if srci == src and dsti == dst:
                        return has_travelled_bw
                        has_t = 0
                        for j in range(i,len(path)):
                            has_t = has_t + path[j][2]
                        return has_t
                    has_travelled_bw = has_travelled_bw + self.latency(srci,dsti,0)
            assert False, f'missing {c} in {src} -> {dst}'

        # Populate data for heuristic = 5
        # Backtrack sends from the final send at the solution-obtained time
        def dst_new_pos(new_pos,src_next,r_next,ii,pi,c):
            for (t_dst, c_dst, p_dst) in new_pos[(src_next,r_next,ii)]:
                if p_dst == pi:
                    assert c_dst == c
                    return t_dst - self.latency(src_next,r_next,0)
            assert False, "shouldn't reach here"
        def get_last_pos():
            pos_last = defaultdict(list)
            new_pos_last = defaultdict(list)
            max_len = 0
            for c in paths:
                for path in paths[c]:
                    if len(path) >= max_len:
                        max_len = len(path)
            path_list = []
            chunk_list = []
            for c in paths:
                for i in range(len(paths[c])):
                    path_list.append(paths[c][i])
                    chunk_list.append(c)
            path_list, chunk_list = zip(*sorted(zip(path_list, chunk_list), key=lambda x: -len(x[0])))

            time = int(self.time[0].X + 0.005)
            for i in range(max_len):
                for j, path in enumerate(path_list):
                    if len(path) > i:
                        src,r,bw = path[i]
                        chunk = chunk_list[j]
                        if i == 0:
                            pos_last[(src,r,i)].append((time, chunk, j))
                        else:
                            pos_last[(src,r,i)].append((-1, chunk, j))
                    else:
                        continue
                for (srci,ri,j) in pos_last:
                    if j == i:
                        lat = self.latency(srci,ri,0)
                        if i == 0:
                            for k, (t,c,p) in enumerate(sorted(pos_last[(srci,ri,i)], key=lambda x: (to_travel(x[1],srci,ri), -has_travelled(x[1],srci,ri)))):
                                new_pos_last[(srci,ri,i)].append((t - k*lat, c, p))
                        else:
                            t_curr = int(self.time[0].X + 0.005) + 10000
                            ii = i-1
                            while ii >= 0:
                                if (srci,ri,ii) in new_pos_last:
                                    for (t_prev,_,_) in new_pos_last[(srci,ri,ii)]:
                                        if t_prev <= t_curr:
                                            t_curr = t_prev
                                ii = ii - 1
                            for k, (t,c,p) in enumerate(sorted(pos_last[(srci,ri,i)], key=lambda x: (to_travel(x[1],srci,ri), -has_travelled(x[1],srci,ri), -dst_new_pos(new_pos_last, path_list[p][i-1][0], path_list[p][i-1][1], i-1, p, chunk_list[p])))):
                                np_last = new_pos_last[(path_list[p][i-1][0],path_list[p][i-1][1],i-1)]
                                # for (tt, src_last, r_last) in np_last:
                                #     if (tt-self.latency(src_last, r_last, 0)) >= t_max:
                                #         t_max = tt-self.latency(src_last, r_last, 0)
                                has_next = False
                                for (t_next, c_next, p_next) in np_last:
                                    if c_next == c and p_next == p:
                                        t_prev = t_next - self.latency(path_list[p_next][i-1][0],path_list[p_next][i-1][1],0)
                                        t_curr = min(t_curr-self.latency(srci,ri,0), t_prev)
                                        new_pos_last[(srci,ri,i)].append((t_curr, c, p))
                                        has_next = True
                                assert has_next
                        # print(i,srci,ri,new_pos_last[(srci,ri,i)])
            return new_pos_last

        pos_last = get_last_pos()
        # for k in pos_last:
        #     print(k, pos_last[k])

        def order_pos_last(c, src, r):
            for (srci,ri,i) in pos_last:
                if srci == src and ri == r:
                    for t, ci, _ in pos_last[src,r,i]:
                        if c == ci:
                            return t

        num_local_nodes = R // self.topology.copies
        # Sort chunk receives and sends according to the heuristics
        # Hardcode NIC groups for DGX2
        # TODO move this elsewhere
        if "DGX2" in self.topology.name:
            nic_groups = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15),(16,17),(18,19),(20,21),(22,23),(24,25),(26,27),(28,29),(30,31)]
            for ni, nic_recv in enumerate(nic_groups):
                for nic_send in nic_groups:
                    if nic_send[0]//num_local_nodes  != nic_recv[0]//num_local_nodes:
                        for r1 in nic_recv:
                            for src in nic_send:
                                l = 0
                                nic_time_recv[ni][l].extend(zip(time_recv[r1][src][l], [src]*len(time_recv[r1][src][l]), [r1]*len(time_recv[r1][src][l])))
                                nic_chunk_recv[ni][l].extend(zip(chunk_recv[r1][src][l], [src]*len(chunk_recv[r1][src][l]), [r1]*len(chunk_recv[r1][src][l])))
                if len(nic_chunk_recv[ni][l]):
                    if heuristic == 0:
                        nic_time_recv[ni][l], nic_chunk_recv[ni][l] = zip(*sorted(zip(nic_time_recv[ni][l], nic_chunk_recv[ni][l]), key=lambda x: (x[0][0], (x[0][1]-x[0][2]+R)%R, x[0][2])))
                    elif heuristic == 1:
                        nic_time_recv[ni][l], nic_chunk_recv[ni][l] = zip(*sorted(zip(nic_time_recv[ni][l], nic_chunk_recv[ni][l]), key=lambda x: (-to_travel(x[1][0],x[0][1],x[0][2]), has_travelled(x[1][0],x[0][1],x[0][2]), (x[0][1]-x[0][2] + R)%R, x[0][2])))
                    elif heuristic == 5:
                        nic_time_recv[ni][l], nic_chunk_recv[ni][l] = zip(*sorted(zip(nic_time_recv[ni][l], nic_chunk_recv[ni][l]), key=lambda x: (x[0][0], order_pos_last(x[1][0],x[0][1],x[0][2]), (x[0][1]-x[0][2] + R)%R, x[0][2])))
                    else:
                        nic_time_recv[ni][l], nic_chunk_recv[ni][l] = zip(*sorted(zip(nic_time_recv[ni][l], nic_chunk_recv[ni][l]), key=lambda x: (x[0][0], -to_travel(x[1][0],x[0][1],x[0][2]), has_travelled(x[1][0],x[0][1],x[0][2]), (x[0][1]-x[0][2] + R)%R, x[0][2])))
                    print("nic_chunk_recv", ni)
                    print(nic_time_recv[ni][l])
                    for i in range(len(nic_chunk_recv[ni][l])):
                        c_recv = nic_chunk_recv[ni][l][i][0]
                        s_recv = nic_chunk_recv[ni][l][i][1]
                        r_recv = nic_chunk_recv[ni][l][i][2]
                        t_recv = nic_time_recv[ni][l][i][0]
                        tup = (i, s_recv, r_recv, c_recv, t_recv, order_pos_last(c_recv,s_recv,r_recv),(s_recv-r_recv + R)%R)
                        print(tup, end=",")
                    print("")

            for ni, nic_send in enumerate(nic_groups):
                for nic_recv in nic_groups:
                    if nic_recv[0]//num_local_nodes  != nic_send[0]//num_local_nodes:
                        for r1 in nic_send:
                            for dst in nic_recv:
                                l = 0
                                nic_time_send[ni][l].extend(zip(time_send[r1][dst][l], [dst]*len(time_send[r1][dst][l]), [r1]*len(time_send[r1][dst][l])))
                                nic_chunk_send[ni][l].extend(zip(chunk_send[r1][dst][l], [dst]*len(chunk_send[r1][dst][l]), [r1]*len(chunk_send[r1][dst][l])))
                if len(nic_chunk_send[ni][l]):
                    if heuristic == 0:
                        nic_time_send[ni][l], nic_chunk_send[ni][l] = zip(*sorted(zip(nic_time_send[ni][l], nic_chunk_send[ni][l]), key=lambda x: (x[0][0], (x[0][2]-x[0][1]+R)%R, x[0][1])))
                    elif heuristic == 1:
                        nic_time_send[ni][l], nic_chunk_send[ni][l] = zip(*sorted(zip(nic_time_send[ni][l], nic_chunk_send[ni][l]), key=lambda x: (-to_travel(x[1][0],x[0][2],x[0][1]), has_travelled(x[1][0],x[0][2],x[0][1]), (x[0][2]-x[0][1] + R)%R, x[0][1])))
                    elif heuristic == 5:
                        nic_time_send[ni][l], nic_chunk_send[ni][l] = zip(*sorted(zip(nic_time_send[ni][l], nic_chunk_send[ni][l]), key=lambda x: (x[0][0], order_pos_last(x[1][0],x[0][2],x[0][1]), (x[0][2]-x[0][1] + R)%R, x[0][1])))
                    else:
                        nic_time_send[ni][l], nic_chunk_send[ni][l] = zip(*sorted(zip(nic_time_send[ni][l], nic_chunk_send[ni][l]), key=lambda x: (x[0][0], -to_travel(x[1][0],x[0][2],x[0][1]), has_travelled(x[1][0],x[0][2],x[0][1]), (x[0][2]-x[0][1] + R)%R, x[0][1])))


        # TODO be able to differentiate between two switches betwen same two nodes, and two switches with the same source node / dst node
        # Right now, we assume that there will never be two switches between same two nodes (okay for DGX2Single but fails for DGX2)
        for r1 in range(R): # dst for recv, source for send
            ll = 0
            sw_added = []
            for r2 in range(R): # source for recv, dst for send
                if (r2,r1) in self.topology.switches_involved:
                    l = 0
                    # Assumes all r2 to r1 connections have same l
                    for swt_i in self.topology.switches_involved[(r2,r1)]:
                        if swt_i not in sw_added:
                            for srcs, dsts, _, _, switch_name in self.topology.switches[swt_i]:
                                if r1 in dsts and "in" in switch_name:
                                    print("switch: r,srcs",r1,srcs)
                                    for src in srcs:
                                        if len(time_recv[r1][src][l]):
                                            switch_time_recv[r1][ll].extend(zip(time_recv[r1][src][l], [src]*len(time_recv[r1][src][l])))
                                            switch_chunk_recv[r1][ll].extend(zip(chunk_recv[r1][src][l], [src]*len(chunk_recv[r1][src][l])))
                                    # TODO need a heuristic which ensures that nic group transfers are ordered together
                                    if len(switch_chunk_recv[r1][ll]):
                                        if heuristic == 0:
                                            switch_time_recv[r1][ll], switch_chunk_recv[r1][ll] = zip(*sorted(zip(switch_time_recv[r1][ll], switch_chunk_recv[r1][ll]), key=lambda x: (x[0][0], (x[0][1]-r1+R)%R)))
                                        elif heuristic == 1:
                                            switch_time_recv[r1][ll], switch_chunk_recv[r1][ll] = zip(*sorted(zip(switch_time_recv[r1][ll], switch_chunk_recv[r1][ll]), key=lambda x: (-to_travel(x[1][0],x[0][1],r1), has_travelled(x[1][0],x[0][1],r1), (x[0][1]-r1 + R)%R)))
                                        elif heuristic == 5:
                                            switch_time_recv[r1][ll], switch_chunk_recv[r1][ll] = zip(*sorted(zip(switch_time_recv[r1][ll], switch_chunk_recv[r1][ll]), key=lambda x: (x[0][0], order_pos_last(x[1][0],x[0][1],r1), (x[0][1]-r1 + R)%R)))
                                        else:
                                            switch_time_recv[r1][ll], switch_chunk_recv[r1][ll] = zip(*sorted(zip(switch_time_recv[r1][ll], switch_chunk_recv[r1][ll]), key=lambda x: (x[0][0], -to_travel(x[1][0],x[0][1],r1), has_travelled(x[1][0],x[0][1],r1), (x[0][1]-r1 + R)%R)))
                                        print("switch_chunk_recv", r1)
                                        print(switch_time_recv[r1][ll])
                                        for i in range(len(switch_chunk_recv[r1][ll])):
                                            c_recv = switch_chunk_recv[r1][ll][i][0]
                                            s_recv = switch_chunk_recv[r1][ll][i][1]
                                            t_recv = switch_time_recv[r1][ll][i][0]
                                            tup = (i, s_recv, c_recv, t_recv, order_pos_last(c_recv,s_recv,r1),(s_recv-r1 + R)%R)
                                            print(tup, end=",")
                                        print("")
                                    l = l + 1
                                    ll = ll + 1
                                    break
                            sw_added.append(swt_i)
        for r1 in range(R): # dst for recv, source for send
            ll = 0
            sw_added = []
            for r2 in range(R): # source for recv, dst for send
                if (r1,r2) in self.topology.switches_involved:
                    l = 0
                    # Assumes all r1 to r2 connections have same l
                    for swt_i in self.topology.switches_involved[(r1,r2)]:
                        if swt_i not in sw_added:
                            for srcs, dsts, _, _, switch_name in self.topology.switches[swt_i]:
                                if r1 in srcs and "out" in switch_name:
                                    for dst in dsts:
                                        if len(time_send[r1][dst][l]):
                                            switch_time_send[r1][ll].extend(zip(time_send[r1][dst][l], [dst]*len(time_send[r1][dst][l])))
                                            switch_chunk_send[r1][ll].extend(zip(chunk_send[r1][dst][l], [dst]*len(chunk_send[r1][dst][l])))
                                    # TODO need a heuristic which ensures that nic group transfers are ordered together
                                    if len(switch_chunk_send[r1][ll]):
                                        if heuristic == 0:
                                            switch_time_send[r1][ll], switch_chunk_send[r1][ll] = zip(*sorted(zip(switch_time_send[r1][ll], switch_chunk_send[r1][ll]), key=lambda x: (x[0][0], (r1-x[0][1]+R)%R)))
                                        elif heuristic == 1:
                                            switch_time_send[r1][ll], switch_chunk_send[r1][ll] = zip(*sorted(zip(switch_time_send[r1][ll], switch_chunk_send[r1][ll]), key=lambda x: (-to_travel(x[1][0],r1,x[0][1]), has_travelled(x[1][0],r1,x[0][1]), (r1-x[0][1]+R)%R)))
                                        elif heuristic == 5:
                                            switch_time_send[r1][ll], switch_chunk_send[r1][ll] = zip(*sorted(zip(switch_time_send[r1][ll], switch_chunk_send[r1][ll]), key=lambda x: (x[0][0], order_pos_last(x[1][0],r1,x[0][1]), (r1-x[0][1]+R)%R)))
                                        else:
                                            switch_time_send[r1][ll], switch_chunk_send[r1][ll] = zip(*sorted(zip(switch_time_send[r1][ll], switch_chunk_send[r1][ll]), key=lambda x: (x[0][0], -to_travel(x[1][0],r1,x[0][1]), has_travelled(x[1][0],r1,x[0][1]), (r1-x[0][1]+R)%R)))
                                        print("switch_chunk_send", r1, *zip(list(range(len(switch_chunk_send[r1][ll]))),switch_chunk_send[r1][ll]))
                                    l = l + 1
                                    ll = ll + 1
                                    break
                            sw_added.append(swt_i)
        for r1 in range(R): # source
            for r2 in range(R): #dst
                for l in range(L):
                    # Sort chunks and times for each rank and link
                    if (len(time_send[r1][r2][l])):
                        # r1: source, r2: dst
                        if heuristic == 0:
                            time_send[r1][r2][l], chunk_send[r1][r2][l] = zip(*sorted(zip(time_send[r1][r2][l], chunk_send[r1][r2][l])))
                        elif heuristic == 1:
                            time_send[r1][r2][l], chunk_send[r1][r2][l] = zip(*sorted(zip(time_send[r1][r2][l], chunk_send[r1][r2][l]), key=lambda x: (-to_travel(x[1],r1,r2), has_travelled(x[1],r1,r2))))
                        elif heuristic == 5:
                            time_send[r1][r2][l], chunk_send[r1][r2][l] = zip(*sorted(zip(time_send[r1][r2][l], chunk_send[r1][r2][l]), key=lambda x: (x[0], order_pos_last(x[1],r1,r2), (r1-r2 + R)%R)))
                        else:
                            time_send[r1][r2][l], chunk_send[r1][r2][l] = zip(*sorted(zip(time_send[r1][r2][l], chunk_send[r1][r2][l]), key=lambda x: (x[0], -to_travel(x[1],r1,r2), has_travelled(x[1],r1,r2))))
                    if (len(time_recv[r2][r1][l])):
                        # r1: source, r2: dst
                        if heuristic == 0:
                            time_recv[r2][r1][l], chunk_recv[r2][r1][l] = zip(*sorted(zip(time_recv[r2][r1][l], chunk_recv[r2][r1][l])))
                        elif heuristic == 1:
                            time_recv[r2][r1][l], chunk_recv[r2][r1][l] = zip(*sorted(zip(time_recv[r2][r1][l], chunk_recv[r2][r1][l]), key=lambda x: (-to_travel(x[1],r1,r2), has_travelled(x[1],r1,r2))))
                        elif heuristic == 5:
                            time_recv[r2][r1][l], chunk_recv[r2][r1][l] = zip(*sorted(zip(time_recv[r2][r1][l], chunk_recv[r2][r1][l]), key=lambda x: (x[0], order_pos_last(x[1],r1,r2), (r1-r2 + R)%R)))
                        else:
                            time_recv[r2][r1][l], chunk_recv[r2][r1][l] = zip(*sorted(zip(time_recv[r2][r1][l], chunk_recv[r2][r1][l]), key=lambda x: (x[0], -to_travel(x[1],r1,r2), has_travelled(x[1],r1,r2))))
                    chunk_send[r1][r2][l] = np.array(chunk_send[r1][r2][l])
                    chunk_recv[r2][r1][l] = np.array(chunk_recv[r2][r1][l])
                    time_send[r1][r2][l] = np.array(time_send[r1][r2][l])
                    time_recv[r2][r1][l] = np.array(time_recv[r2][r1][l])

        with open(f'solutions/solution_{instance_name}_tmp.pkl', 'wb') as f:
            pkl.dump([time_recv, chunk_recv, time_send, chunk_send, model_str, end_time-start_time], f)
        # passed = self._sanity_check(time_recv, chunk_recv, time_send, chunk_send, self.time[0].X)
        os.rename(f'solutions/solution_{instance_name}_tmp.pkl', f'solutions/solution_{instance_name}.pkl')
        return [time_recv, chunk_recv, switch_time_recv, switch_chunk_recv, switch_time_send, switch_chunk_send, nic_time_recv, nic_chunk_recv, nic_time_send, nic_chunk_send]


    # [Deprecated] Used to seed the encoding from a previous lower chunkup
    def optimize_double(self, chunkup=1):
        self.collective = self.collective_og.chunk_up(chunkup)
        self.spsets = shortest_path_sets(self.topology, self.collective)
        instance_name = 'sccl_{}_{}_gurobiSimple'.format(self.topology.name, self.collective.name)

        start_time = time()
        opt = Model(instance_name)
        self._encode(opt, 0)
        opt.optimize()
        end_time = time()

        if opt.status == GRB.INFEASIBLE:
            opt.computeIIS()
            opt.write(f'model_{instance_name}.ilp')
            raise ValueError("Infeasible model")

        C = self.collective.num_chunks
        R = self.collective.num_nodes
        L = 0
        for src in range(R):
            for dst in self.topology.destinations(src):
                if self.topology.link(src,dst) > L:
                    L = self.topology.link(src,dst)

        time_recv = [[[[] for l in range(L)] for src in range(R)] for r in range(R)]
        chunk_recv = [[[[] for l in range(L)] for src in range(R)] for r in range(R)]
        time_send = [[[[] for l in range(L)] for src in range(R)] for r in range(R)]
        chunk_send = [[[[] for l in range(L)] for src in range(R)] for r in range(R)]

        def argsort(s,p):
            indexes = s.argsort()
            return s[indexes], p[indexes]

        model_str = ""
        for c in range(C):
            for r in range(R):
                if self.start[0][c,r].X <= 48:
                    model_str += f'start[{c},{r}]={self.start[0][c,r].X}\n'
        # incoming_nodes = [{1} for r in range(R)]
        for c in range(C):
            for r in range(R):
                for src in self.topology.sources(r):
                    for l in range(L):
                        if self.is_sent[0][c,src,r,l].X >= 0.995:
                            model_str += f'{c}: {src} --{l}--> {r}  t={self.send[0][c,src,r,l].X}\n'
                            chunk_send[src][r][l].append(c)
                            time_send[src][r][l].append(int(self.send[0][c,src,r,l].X + 0.005))
                            # time_send[src][r][l].append(self.send[0][c,src,r,l].X)
                            chunk_recv[r][src][l].append(c)
                            time_recv[r][src][l].append(int(self.start[0][c,r].X + 0.005))
                            # time_recv[r][src][l].append(self.start[0][c,r].X)

        for r in range(R):
            for src in range(R):
                for l in range(L):
                    # Sort chunks and times for each rank and link
                    if (len(time_send[r][src][l])):
                        time_send[r][src][l], chunk_send[r][src][l] = zip(*sorted(zip(time_send[r][src][l], chunk_send[r][src][l])))
                    if (len(time_recv[r][src][l])):
                        time_recv[r][src][l], chunk_recv[r][src][l] = zip(*sorted(zip(time_recv[r][src][l], chunk_recv[r][src][l])))
                    # time_send[r][src][l], chunk_send[r][src][l] = argsort(time_send[r][src][l], chunk_send[r][src][l])
                    # time_recv[r][src][l], chunk_recv[r][src][l] = argsort(time_recv[r][src][l], chunk_recv[r][src][l])
                    chunk_send[r][src][l] = np.array(chunk_send[r][src][l])
                    chunk_recv[r][src][l] = np.array(chunk_recv[r][src][l])
                    time_send[r][src][l] = np.array(time_send[r][src][l])
                    time_recv[r][src][l] = np.array(time_recv[r][src][l])

        with open(f'solutions/solution_{instance_name}_tmp.pkl', 'wb') as f:
            pkl.dump([time_recv, chunk_recv, time_send, chunk_send, model_str, end_time-start_time], f)
        passed = self._sanity_check(time_recv, chunk_recv, time_send, chunk_send, self.time[0].X)
        print(model_str)
        print("Sanity check passed:", passed)
        os.rename(f'solutions/solution_{instance_name}_tmp.pkl', f'solutions/solution_{instance_name}.pkl')

        opt1 = Model(instance_name)
        self._encode(opt1, 1)
        relay_send_order = [6,7,5,2,3,4,1] # sorted by farthest distance from relay in 3-rings
        copies = self.topology.copies
        num_nodes = self.topology.num_nodes()
        num_chunks = self.collective.num_chunks
        num_local_nodes = self.topology.num_nodes() // copies
        num_local_chunks = self.collective.num_chunks // copies
        chunk_per_node = num_local_chunks // num_local_nodes

        for r1 in range(num_local_nodes):
            for i in range(1,copies):
                for j, dst_local_after in enumerate(relay_send_order):
                    c_after = (dst_local_after + i*num_local_nodes) * chunk_per_node + r1
                    assert self.collective.precondition(r1, c_after) and self.collective.postcondition(dst_local_after + i*num_local_nodes, c_after)
                    relay_dst = i*num_local_nodes
                    if j == 0:
                        # opt.addLConstr(self.is_sent[iid][c_after,0,relay_dst,0] == 1)
                        self.is_sent[1][c_after,0,relay_dst,0].start = 1
                        self.send[1][c_after,0,relay_dst,0].start = self.send[0][c_after,0,relay_dst,0].X
                        continue
                    # for dst_local_before in relay_send_order[j-1]:
                    dst_local_before = relay_send_order[j-1]
                    c_before = (dst_local_before + i*num_local_nodes) * chunk_per_node + r1
                    assert self.collective.precondition(r1, c_before) and self.collective.postcondition(dst_local_before + i*num_local_nodes, c_before)
                    # opt.addLConstr(self.is_sent[iid][c_after,0,relay_dst,0] == 1)
                    self.is_sent[1][c_after,0,relay_dst,0].start = 1
                    self.send[1][c_after,0,relay_dst,0].start = self.send[0][c_after,0,relay_dst,0].X
                    # opt.addLConstr(self.send[iid][c_after,0,relay_dst,0] >= self.send[iid][c_before,0,relay_dst,0] + 1)
                    # self.send[iid][c_after,0,relay_dst,0].start >= self.send[iid][c_before,0,relay_dst,0].start

        opt1.optimize()
        return [time_recv, chunk_recv, time_send, chunk_send]
