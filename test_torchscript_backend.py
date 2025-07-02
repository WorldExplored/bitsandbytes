#!/usr/bin/env python3
"""
Test script for TorchScript 4-bit dequantization backend.
"""

import torch
import sys
import os

# Add the bitsandbytes directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import bitsandbytes as bnb
import bitsandbytes.functional as F

def test_torchscript_nf4_dequantization():
    """Test TorchScript NF4 dequantization against default backend."""
    print("Testing TorchScript NF4 dequantization...")
    
    # Create test data
    torch.manual_seed(42)
    original_data = torch.randn(128, 256, dtype=torch.float16)
    
    # Quantize with NF4
    quantized, quant_state = F.quantize_nf4(original_data, blocksize=64)
    print(f"Original shape: {original_data.shape}")
    print(f"Quantized shape: {quantized.shape}")
    print(f"Quant state blocksize: {quant_state.blocksize}")
    print(f"Quant state type: {quant_state.quant_type}")
    
    # Test 1: Default backend dequantization (CPU)
    try:
        dequantized_default = F.dequantize_nf4(quantized, quant_state)
        print(f"Default backend dequantization successful: {dequantized_default.shape}")
    except Exception as e:
        print(f"Default backend failed: {e}")
        return False
    
    # Test 2: Direct TorchScript backend call
    try:
        from bitsandbytes.backends.torchscript.ops import torchscript_dequantize_4bit
        
        dequantized_torchscript = torchscript_dequantize_4bit(
            quantized, 
            quant_state.absmax, 
            quant_state.blocksize, 
            quant_state.quant_type, 
            quant_state.shape, 
            quant_state.dtype
        )
        print(f"TorchScript backend dequantization successful: {dequantized_torchscript.shape}")
        
        # Compare results
        diff = torch.abs(dequantized_default - dequantized_torchscript).max()
        print(f"Maximum difference between backends: {diff}")
        
        if diff < 1e-5:
            print("✅ TorchScript backend produces identical results!")
            return True
        else:
            print(f"❌ Results differ by {diff}")
            return False
            
    except Exception as e:
        print(f"TorchScript backend failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_linear4bit_cpu():
    """Test Linear4bit module on CPU to verify dispatch works."""
    print("\nTesting Linear4bit on CPU...")
    
    try:
        # Create a Linear4bit layer
        layer = bnb.nn.LinearNF4(64, 32)
        
        # Move to CPU to ensure we use non-CUDA path
        layer = layer.cpu()
        
        # Create input
        x = torch.randn(8, 64, dtype=torch.float16)
        
        # Forward pass
        output = layer(x)
        print(f"Linear4bit forward pass successful: {output.shape}")
        return True
        
    except Exception as e:
        print(f"Linear4bit forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing TorchScript 4-bit backend implementation...")
    print("=" * 60)
    
    # Set device to CPU to avoid CUDA path
    torch.set_default_device("cpu")
    
    test1_passed = test_torchscript_nf4_dequantization()
    test2_passed = test_linear4bit_cpu()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("✅ All tests passed! TorchScript backend is working correctly.")
    else:
        print("❌ Some tests failed. Check implementation.")
        sys.exit(1) 