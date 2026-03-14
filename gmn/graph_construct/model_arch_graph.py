# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import torch.nn as nn

from gmn.graph_construct.constants import CONV_LAYERS, NORM_LAYERS
from gmn.graph_construct.hash_grid import MultiResHashGrid
from gmn.graph_construct.layers import PositionwiseFeedForward, BasicBlock, SelfAttention, EquivSetLinear, TriplanarGrid, TriplanarGridWithInputEncoding
from gmn.graph_construct.utils import conv_to_graph, linear_to_graph, norm_to_graph, ffn_to_graph, basic_block_to_graph, self_attention_to_graph, equiv_set_linear_to_graph, triplanar_to_graph, hash_grid_to_graph


def sequential_to_arch(model):
    # input can be a nn.Sequential
    # or ordered list of modules
    arch = []
    weight_bias_modules = CONV_LAYERS + [nn.Linear] + NORM_LAYERS
    for module in model:
        layer = [type(module)]
        if type(module) in weight_bias_modules:
            layer.append(module.weight)
            layer.append(module.bias)
        elif type(module) == BasicBlock:
            layer.extend([
                module.conv1.weight,
                module.bn1.weight,
                module.bn1.bias,
                module.conv2.weight,
                module.bn2.weight,
                module.bn2.bias])
            if len(module.shortcut) > 0:
                layer.extend([
                    module.shortcut[0].weight,
                    module.shortcut[1].weight,
                    module.shortcut[1].bias])
        elif type(module) == PositionwiseFeedForward:
            layer.append(module.lin1.weight)
            layer.append(module.lin1.bias)
            layer.append(module.lin2.weight)
            layer.append(module.lin2.bias)
        elif type(module) == SelfAttention:
            layer.append(module.attn.in_proj_weight)
            layer.append(module.attn.in_proj_bias)
            layer.append(module.attn.out_proj.weight)
            layer.append(module.attn.out_proj.bias)
        elif type(module) == EquivSetLinear:
            layer.append(module.lin1.weight)
            layer.append(module.lin1.bias)
            layer.append(module.lin2.weight)
        elif type(module) == TriplanarGrid:
            layer.append(module.tgrid)
            layer.append(3)
        elif type(module) == TriplanarGridWithInputEncoding:
            layer.append(module.tgrid)
            layer.append(module.encoded_in_dim)
        elif type(module) == MultiResHashGrid:
            layer.append(module.tensorize())
        else:
            if len(list(module.parameters())) != 0:
                raise ValueError(f"{type(module)} has parameters but is not yet supported")
            continue
        arch.append(layer)
    return arch

def arch_to_graph(arch, self_loops=False):
    curr_idx = 0
    x = []
    edge_index = []
    edge_attr = []
    layer_num = 0
    
    # initialize input nodes
    layer = arch[0]
    if layer[0] in CONV_LAYERS:
        in_neuron_idx = torch.arange(layer[1].shape[1])
    elif layer[0] in (nn.Linear, PositionwiseFeedForward):
        in_neuron_idx = torch.arange(layer[1].shape[1])
    elif layer[0] == BasicBlock:
        in_neuron_idx = torch.arange(layer[1].shape[1])
    elif layer[0] == EquivSetLinear:
        in_neuron_idx = torch.arange(layer[1].shape[1])
    elif layer[0] in (TriplanarGrid, TriplanarGridWithInputEncoding):
        triplanar_resolution = layer[1].shape[2]
        in_neuron_idx = torch.arange(3*triplanar_resolution**2)
    elif layer[0] == MultiResHashGrid:
        n_levels = layer[1].shape[0]
        max_tab_len = layer[1].shape[1]
        in_neuron_idx = torch.arange(n_levels * max_tab_len)
    else:
        raise ValueError("Invalid first layer")
    
    for i, layer in enumerate(arch):
        out_neuron = (i==len(arch)-1)
        if layer[0] in CONV_LAYERS:
            ret = conv_to_graph(layer[1], layer[2], layer_num, in_neuron_idx, out_neuron, curr_idx, self_loops)
            layer_num += 1
        elif layer[0] == nn.Linear:
            ret = linear_to_graph(layer[1], layer[2], layer_num, in_neuron_idx, out_neuron, curr_idx, self_loops)
            layer_num += 1
        elif layer[0] in NORM_LAYERS:
            if layer[0] in (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d):
                norm_type = "bn"
            elif layer[0] == nn.LayerNorm:
                norm_type = "ln"
            elif layer[0] == nn.GroupNorm:
                norm_type = "gn"
            elif layer[0] in (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d):
                norm_type = "in"
            else:
                raise ValueError("Invalid norm type")
            ret = norm_to_graph(layer[1], layer[2], layer_num, in_neuron_idx, out_neuron, curr_idx, self_loops, norm_type=norm_type)
        elif layer[0] == BasicBlock:
            ret = basic_block_to_graph(layer[1:], layer_num, in_neuron_idx, out_neuron, curr_idx, self_loops)
            layer_num += 2
        elif layer[0] == PositionwiseFeedForward:
            ret = ffn_to_graph(layer[1], layer[2], layer[3], layer[4], layer_num, in_neuron_idx, out_neuron, curr_idx, self_loops)
            layer_num += 2
        elif layer[0] == SelfAttention:
            ret = self_attention_to_graph(layer[1], layer[2], layer[3], layer[4], layer_num, in_neuron_idx, out_neuron=out_neuron, curr_idx=curr_idx, self_loops=self_loops)
            layer_num += 2
        elif layer[0] == EquivSetLinear:
            ret = equiv_set_linear_to_graph(layer[1], layer[2], layer[3], layer_num, in_neuron_idx, out_neuron=out_neuron, curr_idx=curr_idx, self_loops=self_loops)
            layer_num += 1
        elif layer[0] in (TriplanarGrid, TriplanarGridWithInputEncoding):
            ret = triplanar_to_graph(layer[1], layer[2], layer_num, out_neuron=out_neuron, curr_idx=curr_idx)
            layer_num += 1
        elif layer[0] == MultiResHashGrid:
            ret = hash_grid_to_graph(layer[1], layer_num)
        else:
            raise ValueError("Invalid layer type")
        in_neuron_idx = ret["out_neuron_idx"]
            
        edge_index.append(ret["edge_index"])
        edge_attr.append(ret["edge_attr"])
        if ret["added_x"] is not None:
            feat = ret["added_x"]
            x.append(feat)
            curr_idx += feat.shape[0]

    x = torch.cat(x, dim=0)
    edge_index = torch.cat(edge_index, dim=1)
    edge_attr = torch.cat(edge_attr, dim=0)
    return x, edge_index, edge_attr
