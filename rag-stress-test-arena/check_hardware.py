import psutil
import platform

def get_hardware_report():
    print("--- System Hardware Report ---")
    
    # OS Information
    print(f"OS: {platform.system()} {platform.release()} (Version: {platform.version()})")
    
    # CPU Information
    print(f"Processor: {platform.processor()}")
    # Using psutil for core counts
    print(f"Physical Cores: {psutil.cpu_count(logical=False)}")
    print(f"Total Threads: {psutil.cpu_count(logical=True)}")
    
    # RAM Information
    svmem = psutil.virtual_memory()
    total_gb = round(svmem.total / (1024**3), 2)
    print(f"Total RAM: {total_gb} GB")
    
    # Disk Info (C: Drive)
    usage = psutil.disk_usage('C:\\')
    print(f"Disk Capacity: {round(usage.total / (1024**3), 2)} GB")

if __name__ == "__main__":
    get_hardware_report()