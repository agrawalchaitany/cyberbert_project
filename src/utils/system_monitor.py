import os
import psutil
import time
import platform
import threading
import logging
from typing import Dict, Any, Optional, List, Callable
import gc

# Try to import GPU libraries, but handle gracefully if not available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

class SystemMonitor:
    """
    Production-ready system monitoring for ML workloads
    Tracks CPU, memory, GPU, and disk usage during model training and inference
    """
    
    def __init__(self, 
                metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                interval: float = 5.0,
                logger: Optional[logging.Logger] = None):
        """
        Initialize the system monitor
        
        Args:
            metrics_callback: Callback function to receive metrics updates
            interval: Monitoring interval in seconds
            logger: Optional logger instance
        """
        self.metrics_callback = metrics_callback
        self.interval = interval
        self.logger = logger or logging.getLogger(__name__)
        
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        self.last_metrics = {}
        
    def start(self) -> None:
        """Start the monitoring thread"""
        if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
            self.logger.warning("Monitoring thread is already running")
            return
            
        self.stop_event.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True  # Make daemon so it doesn't block program exit
        )
        self.monitoring_thread.start()
        self.logger.debug("System monitoring started")
        
    def stop(self) -> None:
        """Stop the monitoring thread"""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            return
            
        self.stop_event.set()
        self.monitoring_thread.join(timeout=2.0)  # Wait up to 2 seconds
        
        if self.monitoring_thread.is_alive():
            self.logger.warning("Monitoring thread did not terminate gracefully")
        else:
            self.logger.debug("System monitoring stopped")
            
        self.monitoring_thread = None
        
    def _monitoring_loop(self) -> None:
        """Internal monitoring loop that runs in a separate thread"""
        while not self.stop_event.is_set():
            try:
                # Collect metrics
                metrics = self.get_system_metrics()
                
                # Store last metrics
                self.last_metrics = metrics
                
                # Call the callback if provided
                if self.metrics_callback:
                    self.metrics_callback(metrics)
                    
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                
            # Wait for the next interval or until stop is called
            self.stop_event.wait(self.interval)
            
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get current system metrics
        
        Returns:
            Dictionary of system metrics
        """
        metrics = {
            "timestamp": time.time(),
            "cpu": self._get_cpu_metrics(),
            "memory": self._get_memory_metrics(),
            "disk": self._get_disk_metrics(),
        }
        
        # Add GPU metrics if available
        gpu_metrics = self._get_gpu_metrics()
        if gpu_metrics:
            metrics["gpu"] = gpu_metrics
            
        return metrics
        
    def _get_cpu_metrics(self) -> Dict[str, Any]:
        """Get CPU metrics"""
        cpu_percent = psutil.cpu_percent(interval=0.5, percpu=True)
        cpu_freq = psutil.cpu_freq() if hasattr(psutil, 'cpu_freq') and psutil.cpu_freq() else None
        
        metrics = {
            "percent_per_core": cpu_percent,
            "percent_overall": sum(cpu_percent) / len(cpu_percent),
            "count_physical": psutil.cpu_count(logical=False) or 1,
            "count_logical": psutil.cpu_count(logical=True) or 1,
        }
        
        # Add CPU frequency if available
        if cpu_freq:
            metrics["freq_current_mhz"] = cpu_freq.current
            if hasattr(cpu_freq, 'min') and cpu_freq.min:
                metrics["freq_min_mhz"] = cpu_freq.min
            if hasattr(cpu_freq, 'max') and cpu_freq.max:
                metrics["freq_max_mhz"] = cpu_freq.max
                
        # Get CPU temperature if available
        try:
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if temps:
                    # Get the first available temperature reading
                    for device, readings in temps.items():
                        if readings:
                            metrics["temperature_celsius"] = readings[0].current
                            break
        except Exception:
            pass  # Not all systems support temperature readings
            
        return metrics
        
    def _get_memory_metrics(self) -> Dict[str, Any]:
        """Get memory metrics"""
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        
        # Force garbage collection to get accurate memory usage
        gc.collect()
        
        metrics = {
            "total_gb": vm.total / (1024 ** 3),
            "available_gb": vm.available / (1024 ** 3),
            "used_gb": vm.used / (1024 ** 3),
            "percent": vm.percent,
            "swap_total_gb": sm.total / (1024 ** 3),
            "swap_used_gb": sm.used / (1024 ** 3),
            "swap_percent": sm.percent
        }
        
        # Add process-specific memory info
        process = psutil.Process(os.getpid())
        
        # Get memory info for this process
        try:
            process_memory = process.memory_info()
            metrics["process"] = {
                "rss_gb": process_memory.rss / (1024 ** 3),  # Resident Set Size
                "vms_gb": process_memory.vms / (1024 ** 3),  # Virtual Memory Size
            }
            
            # Add peak memory usage if available
            if hasattr(process, 'memory_full_info'):
                memory_full = process.memory_full_info()
                if hasattr(memory_full, 'peak_wset'):
                    metrics["process"]["peak_gb"] = memory_full.peak_wset / (1024 ** 3)
                elif hasattr(memory_full, 'peak'):
                    metrics["process"]["peak_gb"] = memory_full.peak / (1024 ** 3)
        except Exception as e:
            pass  # Some systems might restrict process info access
            
        # Add PyTorch-specific memory info if available
        if HAS_TORCH and torch.cuda.is_available():
            try:
                metrics["torch"] = {
                    "allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
                    "cached_gb": torch.cuda.memory_reserved() / (1024 ** 3),
                    "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
                }
            except Exception:
                pass
                
        return metrics
        
    def _get_disk_metrics(self) -> Dict[str, Any]:
        """Get disk metrics"""
        disk_io = psutil.disk_io_counters() if hasattr(psutil, 'disk_io_counters') else None
        
        metrics = {}
        
        # Get disk usage for the current directory
        try:
            disk_usage = psutil.disk_usage(os.getcwd())
            metrics["current_dir"] = {
                "total_gb": disk_usage.total / (1024 ** 3),
                "used_gb": disk_usage.used / (1024 ** 3),
                "free_gb": disk_usage.free / (1024 ** 3),
                "percent": disk_usage.percent
            }
        except Exception:
            pass
            
        # Add disk IO metrics if available
        if disk_io:
            metrics["io"] = {
                "read_mb": disk_io.read_bytes / (1024 ** 2),
                "write_mb": disk_io.write_bytes / (1024 ** 2),
                "read_count": disk_io.read_count,
                "write_count": disk_io.write_count,
            }
            
        return metrics
        
    def _get_gpu_metrics(self) -> Optional[Dict[str, Any]]:
        """Get GPU metrics if available"""
        metrics = {}
        
        # Try PyTorch first
        if HAS_TORCH and torch.cuda.is_available():
            try:
                device_count = torch.cuda.device_count()
                metrics["count"] = device_count
                metrics["devices"] = []
                
                for i in range(device_count):
                    device_props = torch.cuda.get_device_properties(i)
                    
                    # Get memory info
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
                    memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # GB
                    
                    device_info = {
                        "id": i,
                        "name": device_props.name,
                        "total_memory_gb": device_props.total_memory / (1024 ** 3),
                        "allocated_gb": memory_allocated,
                        "reserved_gb": memory_reserved,
                        "percent_used": (memory_allocated / (device_props.total_memory / (1024 ** 3))) * 100
                    }
                    
                    # Try to get utilization using GPUtil as PyTorch doesn't expose this
                    if HAS_GPUTIL:
                        try:
                            gpu = GPUtil.getGPUs()[i]
                            device_info["utilization_percent"] = gpu.utilization
                            device_info["temperature_celsius"] = gpu.temperature
                        except (IndexError, Exception):
                            pass
                            
                    metrics["devices"].append(device_info)
                    
                return metrics
            except Exception as e:
                self.logger.debug(f"Error getting PyTorch GPU metrics: {str(e)}")
                
        # Fallback to GPUtil
        if HAS_GPUTIL:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    metrics["count"] = len(gpus)
                    metrics["devices"] = []
                    
                    for i, gpu in enumerate(gpus):
                        device_info = {
                            "id": i,
                            "name": gpu.name,
                            "utilization_percent": gpu.utilization,
                            "memory_used_gb": gpu.memoryUsed / 1024,  # Convert from MB to GB
                            "memory_total_gb": gpu.memoryTotal / 1024,  # Convert from MB to GB
                            "temperature_celsius": gpu.temperature,
                            "percent_used": gpu.memoryUtil * 100
                        }
                        metrics["devices"].append(device_info)
                        
                    return metrics
            except Exception as e:
                self.logger.debug(f"Error getting GPUtil metrics: {str(e)}")
                
        # No GPU metrics available
        return None
        
    def get_latest_metrics(self) -> Dict[str, Any]:
        """
        Get the latest collected metrics
        
        Returns:
            Dictionary of latest metrics
        """
        return self.last_metrics.copy() if self.last_metrics else self.get_system_metrics()
        
    def print_summary(self) -> None:
        """Print a summary of the current system metrics"""
        metrics = self.get_system_metrics()
        
        print("\n=== System Metrics Summary ===")
        
        # Print CPU info
        cpu = metrics["cpu"]
        print(f"CPU: {cpu['percent_overall']:.1f}% used across {cpu['count_logical']} logical cores")
        
        # Print memory info
        memory = metrics["memory"]
        print(f"Memory: {memory['used_gb']:.1f} GB / {memory['total_gb']:.1f} GB ({memory['percent']}%)")
        
        if "process" in memory:
            print(f"Process Memory: {memory['process']['rss_gb']:.2f} GB (RSS)")
            
        # Print GPU info if available
        if "gpu" in metrics and metrics["gpu"] and "devices" in metrics["gpu"]:
            for device in metrics["gpu"]["devices"]:
                print(f"GPU {device['id']} ({device['name']}): "
                      f"{device.get('allocated_gb', 0):.1f} GB / {device.get('total_memory_gb', 0):.1f} GB "
                      f"({device.get('percent_used', 0):.1f}%)")
                if "utilization_percent" in device:
                    print(f"  - Utilization: {device['utilization_percent']:.1f}%")
                if "temperature_celsius" in device:
                    print(f"  - Temperature: {device['temperature_celsius']:.1f}°C")
                    
        # Print disk info
        if "current_dir" in metrics["disk"]:
            disk = metrics["disk"]["current_dir"]
            print(f"Disk (Current Dir): {disk['used_gb']:.1f} GB / {disk['total_gb']:.1f} GB ({disk['percent']}%)")
            
        print("=============================\n")


def monitor_and_log_system(logger: logging.Logger, save_path: Optional[str] = None,
                           interval: float = 30.0) -> SystemMonitor:
    """
    Utility function to quickly setup system monitoring with logging
    
    Args:
        logger: Logger instance to use for logging
        save_path: Path to save metrics snapshots (optional)
        interval: Monitoring interval in seconds
        
    Returns:
        SystemMonitor instance that has been started
    """
    from .metrics import MetricsTracker
    
    # Create metrics tracker if save path provided
    metrics_tracker = None
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        metrics_tracker = MetricsTracker(save_path, logger)
    
    # Define callback function for system metrics
    def log_metrics(metrics: Dict[str, Any]) -> None:
        # Log a summary line
        mem_used = metrics["memory"]["used_gb"]
        mem_total = metrics["memory"]["total_gb"]
        cpu_pct = metrics["cpu"]["percent_overall"]
        
        # Construct GPU info string if available
        gpu_info = ""
        if "gpu" in metrics and metrics["gpu"] and "devices" in metrics["gpu"]:
            for device in metrics["gpu"]["devices"]:
                gpu_used = device.get('percent_used', 0)
                gpu_info += f", GPU {device['id']}: {gpu_used:.1f}%"
        
        logger.info(f"System: CPU {cpu_pct:.1f}%, RAM {mem_used:.1f}/{mem_total:.1f} GB{gpu_info}")
        
        # Track metrics if we have a tracker
        if metrics_tracker:
            metrics_tracker.track_system_metrics(metrics)
            
            # Periodically save metrics
            if "timestamp" in metrics and int(metrics["timestamp"]) % 300 < interval:  # Every ~5 minutes
                metrics_tracker.save_metrics("system_metrics_latest.json")
    
    # Create and start the system monitor
    monitor = SystemMonitor(
        metrics_callback=log_metrics,
        interval=interval,
        logger=logger
    )
    monitor.start()
    
    return monitor