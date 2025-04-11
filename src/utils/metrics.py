import time
import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
from datetime import datetime

class MetricsTracker:
    """
    Metrics tracking system for production CyberBERT deployments
    Handles tracking, storing, and reporting of training and inference metrics
    """
    
    def __init__(self, save_dir: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the metrics tracker
        
        Args:
            save_dir: Directory to save metrics
            logger: Logger instance (optional)
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.logger = logger or logging.getLogger(__name__)
        self.metrics = {
            "training": {},
            "inference": {},
            "system": {
                "start_time": time.time(),
                "timestamps": []
            }
        }
        
    def track_training_batch(self, batch_metrics: Dict[str, Any], step: int) -> None:
        """
        Track metrics for a single training batch
        
        Args:
            batch_metrics: Dictionary of metrics for the batch
            step: Current training step
        """
        timestamp = time.time()
        
        # Ensure training section exists
        if "batches" not in self.metrics["training"]:
            self.metrics["training"]["batches"] = []
        
        # Add timestamp and step to metrics
        metrics_with_meta = {
            "step": step,
            "timestamp": timestamp,
            **batch_metrics
        }
        
        # Store batch metrics
        self.metrics["training"]["batches"].append(metrics_with_meta)
        
        # Also update system timestamps
        self.metrics["system"]["timestamps"].append(timestamp)
        
        # If we have too many batches stored, keep only the last 100
        if len(self.metrics["training"]["batches"]) > 100:
            self.metrics["training"]["batches"] = self.metrics["training"]["batches"][-100:]
    
    def track_training_epoch(self, epoch_metrics: Dict[str, Any], epoch: int) -> None:
        """
        Track metrics for a training epoch
        
        Args:
            epoch_metrics: Dictionary of metrics for the epoch
            epoch: Current epoch number
        """
        timestamp = time.time()
        
        # Ensure epochs section exists
        if "epochs" not in self.metrics["training"]:
            self.metrics["training"]["epochs"] = []
        
        # Add timestamp and epoch to metrics
        metrics_with_meta = {
            "epoch": epoch,
            "timestamp": timestamp,
            **epoch_metrics
        }
        
        # Store epoch metrics
        self.metrics["training"]["epochs"].append(metrics_with_meta)
    
    def track_inference(self, inference_metrics: Dict[str, Any], batch_size: int = 1) -> None:
        """
        Track metrics for inference (prediction)
        
        Args:
            inference_metrics: Dictionary of metrics for inference
            batch_size: Size of the inference batch
        """
        timestamp = time.time()
        
        # Calculate latency if not provided
        if "latency" not in inference_metrics and "start_time" in inference_metrics:
            inference_metrics["latency"] = timestamp - inference_metrics["start_time"]
            del inference_metrics["start_time"]
        
        # Ensure inferences section exists
        if "requests" not in self.metrics["inference"]:
            self.metrics["inference"]["requests"] = []
            self.metrics["inference"]["total_processed"] = 0
            self.metrics["inference"]["latencies"] = []
        
        # Add timestamp to metrics
        metrics_with_meta = {
            "timestamp": timestamp,
            "batch_size": batch_size,
            **inference_metrics
        }
        
        # Store inference metrics
        self.metrics["inference"]["requests"].append(metrics_with_meta)
        self.metrics["inference"]["total_processed"] += batch_size
        
        # Track latency for calculating percentiles
        if "latency" in inference_metrics:
            self.metrics["inference"]["latencies"].append(inference_metrics["latency"])
        
        # If we have too many inferences stored, keep only the last 1000
        if len(self.metrics["inference"]["requests"]) > 1000:
            self.metrics["inference"]["requests"] = self.metrics["inference"]["requests"][-1000:]
    
    def track_system_metrics(self, system_metrics: Dict[str, Any]) -> None:
        """
        Track system metrics (CPU, memory, GPU, etc.)
        
        Args:
            system_metrics: Dictionary of system metrics
        """
        timestamp = time.time()
        
        # Ensure system_stats section exists
        if "stats" not in self.metrics["system"]:
            self.metrics["system"]["stats"] = []
        
        # Add timestamp to metrics
        metrics_with_meta = {
            "timestamp": timestamp,
            **system_metrics
        }
        
        # Store system metrics
        self.metrics["system"]["stats"].append(metrics_with_meta)
        
        # If we have too many system metrics stored, keep only the most recent ones
        if len(self.metrics["system"]["stats"]) > 100:
            self.metrics["system"]["stats"] = self.metrics["system"]["stats"][-100:]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for tracked metrics
        
        Returns:
            Dictionary of summary statistics
        """
        stats = {
            "uptime": time.time() - self.metrics["system"]["start_time"],
            "training": {},
            "inference": {}
        }
        
        # Calculate inference statistics if we have any
        if "inference" in self.metrics and "latencies" in self.metrics["inference"] and self.metrics["inference"]["latencies"]:
            latencies = self.metrics["inference"]["latencies"]
            stats["inference"]["total_requests"] = self.metrics["inference"]["total_processed"]
            stats["inference"]["latency"] = {
                "mean": np.mean(latencies),
                "median": np.median(latencies),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
                "min": np.min(latencies),
                "max": np.max(latencies)
            }
            
            # Calculate requests per second (throughput) over the last minute
            if "requests" in self.metrics["inference"] and self.metrics["inference"]["requests"]:
                recent_requests = [r for r in self.metrics["inference"]["requests"] 
                                  if r["timestamp"] > time.time() - 60]
                if recent_requests:
                    stats["inference"]["throughput_1min"] = len(recent_requests) / 60
                else:
                    stats["inference"]["throughput_1min"] = 0
        
        # Calculate training statistics if we have epoch data
        if "training" in self.metrics and "epochs" in self.metrics["training"] and self.metrics["training"]["epochs"]:
            epochs = self.metrics["training"]["epochs"]
            latest_epoch = epochs[-1]
            
            stats["training"]["epochs_completed"] = len(epochs)
            stats["training"]["latest_epoch"] = latest_epoch["epoch"]
            
            # Extract metrics that are present in all epochs
            metric_keys = set.intersection(*[set(epoch.keys()) for epoch in epochs]) - {"epoch", "timestamp"}
            
            # Track the metrics over time
            for key in metric_keys:
                if key not in ["epoch", "timestamp"]:
                    values = [epoch[key] for epoch in epochs]
                    stats["training"][key] = {
                        "current": latest_epoch[key],
                        "best": min(values) if "loss" in key.lower() else max(values),
                        "trend": "decreasing" if values[-1] < values[-2] else "increasing" 
                                if len(values) > 1 else "unknown"
                    }
        
        return stats
    
    def save_metrics(self, filename: Optional[str] = None) -> str:
        """
        Save current metrics to a JSON file
        
        Args:
            filename: Optional filename, defaults to metrics_TIMESTAMP.json
            
        Returns:
            Path to the saved metrics file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        file_path = os.path.join(self.save_dir, filename)
        
        # Get statistics to include in the saved metrics
        stats = self.get_stats()
        
        # Prepare metrics for serialization
        serializable_metrics = {
            "summary_stats": stats,
            "raw_metrics": {
                "training": {},
                "inference": {},
                "system": {
                    "start_time": self.metrics["system"]["start_time"],
                }
            }
        }
        
        # Include batches and epochs if they exist
        if "training" in self.metrics:
            if "batches" in self.metrics["training"]:
                serializable_metrics["raw_metrics"]["training"]["recent_batches"] = self.metrics["training"]["batches"]
            if "epochs" in self.metrics["training"]:
                serializable_metrics["raw_metrics"]["training"]["epochs"] = self.metrics["training"]["epochs"]
        
        # Include recent inference requests if they exist
        if "inference" in self.metrics and "requests" in self.metrics["inference"]:
            serializable_metrics["raw_metrics"]["inference"]["recent_requests"] = self.metrics["inference"]["requests"][-50:]  # Only include last 50
        
        # Include system stats if they exist
        if "system" in self.metrics and "stats" in self.metrics["system"]:
            serializable_metrics["raw_metrics"]["system"]["recent_stats"] = self.metrics["system"]["stats"][-20:]  # Only include last 20
        
        try:
            with open(file_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=2, default=str)
            
            if self.logger:
                self.logger.debug(f"Saved metrics to {file_path}")
                
            return file_path
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save metrics: {str(e)}")
            return ""
    
    def get_latest_training_metrics(self) -> Dict[str, Any]:
        """
        Get the latest training metrics for reporting
        
        Returns:
            Dictionary of latest training metrics
        """
        latest = {}
        
        if "training" in self.metrics:
            if "epochs" in self.metrics["training"] and self.metrics["training"]["epochs"]:
                latest["latest_epoch"] = self.metrics["training"]["epochs"][-1]
            
            if "batches" in self.metrics["training"] and self.metrics["training"]["batches"]:
                latest["latest_batch"] = self.metrics["training"]["batches"][-1]
        
        return latest