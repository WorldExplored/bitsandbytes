# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
TorchScript backend for bitsandbytes operations.

This backend provides JIT-compiled fallback implementations for 4-bit quantization
operations on non-CUDA devices (CPU, MPS, etc.).
""" 