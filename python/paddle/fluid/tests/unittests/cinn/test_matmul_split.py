import paddle
import numpy as np
import time
import os
import logging
import sys

# 抑制日志输出
os.environ['GLOG_v'] = '0'
os.environ['GLOG_logtostderr'] = '0'
os.environ['GLOG_minloglevel'] = '2'  # 只显示 ERROR 级别以上的日志
logging.getLogger().setLevel(logging.ERROR)

# 重定向标准错误输出
class NullWriter:
    def write(self, text):
        pass
    def flush(self):
        pass

sys.stderr = NullWriter()

def run_matmul_test(M, K, N, repeat=10):
    paddle.set_device('cpu')
    
    x = paddle.randn([M, K], dtype="float32")
    y = paddle.randn([K, N], dtype="float32")
    
    def matmul_fn(x, y):
        return paddle.matmul(x, y)
    
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = True
    
    input_specs = [
        paddle.static.InputSpec(shape=[M, K], dtype='float32', name='x'),
        paddle.static.InputSpec(shape=[K, N], dtype='float32', name='y')
    ]
    
    compile_fn = paddle.jit.to_static(
        matmul_fn,
        build_strategy=build_strategy,
        input_spec=input_specs,
        full_graph=True
    )
    
    # Warmup
    _ = compile_fn(x, y)
    _ = matmul_fn(x, y)
    paddle.device.cuda.synchronize()  # 确保GPU操作完成
    
    # Test original version
    st = time.time()
    for _ in range(repeat):
        out1 = matmul_fn(x, y)
    paddle.device.cuda.synchronize()
    original_time = time.time() - st
    
    # Test optimized version
    st = time.time() 
    for _ in range(repeat):
        out2 = compile_fn(x, y)
    paddle.device.cuda.synchronize()
    optimized_time = time.time() - st
    
    # Verify results
    try:
        np.testing.assert_allclose(
            out1.numpy(),
            out2.numpy(),
            rtol=1e-5,
            atol=1e-5
        )
    except AssertionError:
        print("Warning: Results mismatch detected!")
    
    return original_time, optimized_time

def test_different_sizes():
    print("\n" + "="*80)
    print("{:^80}".format("Matrix Multiplication Performance Test"))
    print("="*80)
    print("{:<25} {:>15} {:>15} {:>12} {:>12}".format(
        "Matrix Size", 
        "Original(s)", 
        "Optimized(s)", 
        "Speedup", 
        "GFLOPS"
    ))
    print("-"*80)
    
    sizes = [
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 512, 8192),
        (16384, 256, 16384),
        (512, 8192, 512)
    ]
    
    for M, K, N in sizes:
        try:
            original_time, optimized_time = run_matmul_test(M, K, N)
            speedup = original_time / optimized_time
            
            # Calculate GFLOPS
            flops = 2 * M * N * K * 10  # *10 for the number of iterations
            gflops = flops / (optimized_time * 1e9)
            
            size_str = f"{M}x{K}x{N}"
            print("{:<25} {:>15.3f} {:>15.3f} {:>12.2f} {:>12.2f}".format(
                size_str,
                original_time,
                optimized_time,
                speedup,
                gflops
            ))
        except Exception as e:
            print(f"Error testing size {M}x{K}x{N}: {str(e)}")
    
    print("="*80)
    print("Note:")
    print("- Matrix Size (MxK @ KxN): First matrix M rows and K columns, Second matrix K rows and N columns")
    print("- GFLOPS: Billion floating-point operations per second")
    print("- Speedup > 1 indicates performance improvement")
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        test_different_sizes()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
