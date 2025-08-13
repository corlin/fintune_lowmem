import torch
import datetime

def main():
    print(f"GPU测试项目 - 日期: {datetime.datetime.now().strftime('%Y%m%d')}")
    print("=" * 50)
    
    # 检查PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print("CUDA 可用: 是")
        print(f"CUDA 版本: {torch.version.cuda}")
        
        # 获取GPU信息
        device_count = torch.cuda.device_count()
        print(f"检测到 {device_count} 个GPU")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            device_capability = torch.cuda.get_device_capability(i)
            device_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # 转换为GB
            
            print(f"\nGPU {i}:")
            print(f"  名称: {device_name}")
            print(f"  计算能力: {device_capability}")
            print(f"  显存: {device_memory:.2f} GB")
            
        # 测试GPU计算
        print("\n执行GPU计算测试...")
        device = torch.cuda.current_device()
        
        # 创建测试张量
        x = torch.randn(10000, 10000, device=device)
        y = torch.randn(10000, 10000, device=device)
        
        # 执行矩阵乘法
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        z = torch.matmul(x, y)
        end_time.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        
        print(f"矩阵乘法完成，耗时: {elapsed_time:.2f} 毫秒")
        print("GPU测试成功！")
        
    else:
        print("CUDA 可用: 否")
        print("未检测到支持的GPU，将使用CPU进行计算")
        
        # CPU测试
        print("\n执行CPU计算测试...")
        x = torch.randn(5000, 5000)
        y = torch.randn(5000, 5000)
        
        import time
        start_time = time.time()
        z = torch.matmul(x, y)
        end_time = time.time()
        
        elapsed_time = (end_time - start_time) * 1000
        print(f"矩阵乘法完成，耗时: {elapsed_time:.2f} 毫秒")
        print("CPU测试完成！")

if __name__ == "__main__":
    main()
