import os
import json
import time
import uuid
from typing import Dict, Any, Optional
import logging
import hashlib
import torch
from datetime import datetime

class ModelRegistry:
    """
    Production-grade model registry for CyberBERT models.
    Tracks model versions, metadata, and performance metrics.
    Enables model comparison, rollback, and deployment tracking.
    """
    
    def __init__(self, registry_dir: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the model registry
        
        Args:
            registry_dir: Directory to store the model registry
            logger: Logger instance
        """
        self.registry_dir = registry_dir
        self.models_dir = os.path.join(registry_dir, "models")
        self.metadata_file = os.path.join(registry_dir, "model_registry.json")
        self.logger = logger or logging.getLogger(__name__)
        
        # Create registry directories
        os.makedirs(self.registry_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize or load registry
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                "models": {},
                "production_model": None,
                "staging_model": None,
                "development_model": None,
                "last_updated": datetime.now().isoformat()
            }
            self._save_registry()
    
    def _save_registry(self) -> None:
        """Save the registry to disk"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def _compute_model_hash(self, model_path: str) -> str:
        """
        Compute a hash of the model files for integrity verification
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            SHA-256 hash of the model files
        """
        hasher = hashlib.sha256()
        
        for root, _, files in os.walk(model_path):
            for file in sorted(files):  # Sort for deterministic ordering
                file_path = os.path.join(root, file)
                # Skip temporary files
                if file.endswith('.tmp') or file.startswith('.'):
                    continue
                
                # Update hash with file path and content
                rel_path = os.path.relpath(file_path, model_path)
                hasher.update(rel_path.encode())
                
                # For smaller files, hash the entire content
                if os.path.getsize(file_path) < 100 * 1024 * 1024:  # 100MB
                    with open(file_path, 'rb') as f:
                        hasher.update(f.read())
                else:
                    # For large files, hash the first and last 50MB
                    with open(file_path, 'rb') as f:
                        hasher.update(f.read(50 * 1024 * 1024))
                        f.seek(-50 * 1024 * 1024, os.SEEK_END)
                        hasher.update(f.read())
        
        return hasher.hexdigest()
    
    def register_model(self, 
                      model_path: str, 
                      metrics: Dict[str, Any], 
                      metadata: Dict[str, Any],
                      artifacts: Optional[Dict[str, str]] = None) -> str:
        """
        Register a model in the registry
        
        Args:
            model_path: Path to the model directory
            metrics: Dictionary of model performance metrics
            metadata: Dictionary of model metadata
            artifacts: Dictionary of additional artifact paths
            
        Returns:
            Model version ID
        """
        # Generate unique model ID
        model_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Create model directory in registry
        model_registry_dir = os.path.join(self.models_dir, model_id)
        os.makedirs(model_registry_dir, exist_ok=True)
        
        # Compute model hash for integrity verification
        model_hash = self._compute_model_hash(model_path)
        
        # Create model registry entry
        model_entry = {
            "id": model_id,
            "created_at": timestamp,
            "model_path": model_path,
            "registry_path": model_registry_dir,
            "metrics": metrics,
            "metadata": metadata,
            "artifacts": artifacts or {},
            "hash": model_hash,
            "status": "registered",
            "deployment_history": [],
            "tags": []
        }
        
        # Add to registry
        self.registry["models"][model_id] = model_entry
        self.registry["last_updated"] = timestamp
        
        # Update development model pointer
        self.registry["development_model"] = model_id
        
        # Save registry
        self._save_registry()
        
        self.logger.info(f"Registered model {model_id} in registry")
        return model_id
    
    def promote_model(self, model_id: str, environment: str) -> bool:
        """
        Promote a model to a specific environment
        
        Args:
            model_id: Model ID to promote
            environment: Target environment (staging or production)
            
        Returns:
            Success status
        """
        if model_id not in self.registry["models"]:
            self.logger.error(f"Model {model_id} not found in registry")
            return False
            
        if environment not in ["staging", "production"]:
            self.logger.error(f"Invalid environment: {environment}")
            return False
        
        # Update model status
        model = self.registry["models"][model_id]
        prev_status = model["status"]
        model["status"] = environment
        
        # Add to deployment history
        deployment_entry = {
            "timestamp": datetime.now().isoformat(),
            "environment": environment,
            "previous_status": prev_status
        }
        model["deployment_history"].append(deployment_entry)
        
        # Update environment pointer
        env_key = f"{environment}_model"
        prev_model_id = self.registry[env_key]
        self.registry[env_key] = model_id
        
        # If there was a previous model in this environment, update its status
        if prev_model_id and prev_model_id in self.registry["models"]:
            prev_model = self.registry["models"][prev_model_id]
            # If it's not in another environment, mark it as superseded
            if prev_model["status"] == environment:
                prev_model["status"] = "superseded"
                prev_model["deployment_history"].append({
                    "timestamp": datetime.now().isoformat(),
                    "environment": "none",
                    "previous_status": environment,
                    "superseded_by": model_id
                })
        
        # Update last updated timestamp
        self.registry["last_updated"] = datetime.now().isoformat()
        
        # Save registry
        self._save_registry()
        
        self.logger.info(f"Promoted model {model_id} to {environment}")
        return True
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model
        
        Args:
            model_id: Model ID to retrieve
            
        Returns:
            Model information or None if not found
        """
        if model_id in self.registry["models"]:
            return self.registry["models"][model_id]
        return None
    
    def get_latest_model(self, environment: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the latest model for a specific environment
        
        Args:
            environment: Environment (production, staging, development)
            
        Returns:
            Latest model information or None if not found
        """
        if environment is None:
            # Return the most recently created model
            if not self.registry["models"]:
                return None
                
            latest_model_id = max(
                self.registry["models"].keys(),
                key=lambda model_id: self.registry["models"][model_id]["created_at"]
            )
            return self.registry["models"][latest_model_id]
        
        env_key = f"{environment}_model"
        model_id = self.registry.get(env_key)
        
        if model_id and model_id in self.registry["models"]:
            return self.registry["models"][model_id]
        
        return None
    
    def compare_models(self, model_id_1: str, model_id_2: str) -> Dict[str, Any]:
        """
        Compare two models based on their metrics
        
        Args:
            model_id_1: First model ID
            model_id_2: Second model ID
            
        Returns:
            Comparison results
        """
        if model_id_1 not in self.registry["models"] or model_id_2 not in self.registry["models"]:
            self.logger.error("One or both models not found in registry")
            return {"error": "Model not found"}
        
        model1 = self.registry["models"][model_id_1]
        model2 = self.registry["models"][model_id_2]
        
        # Compare metrics
        comparison = {
            "model1": {
                "id": model_id_1,
                "created_at": model1["created_at"],
                "status": model1["status"],
                "metrics": model1["metrics"]
            },
            "model2": {
                "id": model_id_2,
                "created_at": model2["created_at"],
                "status": model2["status"],
                "metrics": model2["metrics"]
            },
            "differences": {}
        }
        
        # Calculate metric differences
        for metric in set(model1["metrics"]) & set(model2["metrics"]):
            if isinstance(model1["metrics"][metric], (int, float)) and isinstance(model2["metrics"][metric], (int, float)):
                diff = model2["metrics"][metric] - model1["metrics"][metric]
                pct_change = diff / model1["metrics"][metric] * 100 if model1["metrics"][metric] != 0 else float('inf')
                
                comparison["differences"][metric] = {
                    "absolute_diff": diff,
                    "percentage_change": pct_change,
                    "better": (pct_change > 0) if "accuracy" in metric.lower() or "f1" in metric.lower() or "precision" in metric.lower() or "recall" in metric.lower() 
                             else (pct_change < 0) if "loss" in metric.lower() or "error" in metric.lower() or "latency" in metric.lower()
                             else None
                }
        
        return comparison
    
    def validate_model_integrity(self, model_id: str) -> bool:
        """
        Validate the integrity of a model by checking its hash
        
        Args:
            model_id: Model ID to validate
            
        Returns:
            True if model is valid, False otherwise
        """
        if model_id not in self.registry["models"]:
            self.logger.error(f"Model {model_id} not found in registry")
            return False
        
        model = self.registry["models"][model_id]
        model_path = model["model_path"]
        
        if not os.path.exists(model_path):
            self.logger.error(f"Model path {model_path} does not exist")
            return False
        
        # Compute current hash
        current_hash = self._compute_model_hash(model_path)
        
        # Compare with stored hash
        if current_hash != model["hash"]:
            self.logger.error(f"Model integrity check failed for {model_id}")
            return False
        
        return True
    
    def add_model_tag(self, model_id: str, tag: str) -> bool:
        """
        Add a tag to a model
        
        Args:
            model_id: Model ID to tag
            tag: Tag to add
            
        Returns:
            Success status
        """
        if model_id not in self.registry["models"]:
            self.logger.error(f"Model {model_id} not found in registry")
            return False
        
        model = self.registry["models"][model_id]
        
        if tag not in model["tags"]:
            model["tags"].append(tag)
            self.registry["last_updated"] = datetime.now().isoformat()
            self._save_registry()
            self.logger.info(f"Added tag '{tag}' to model {model_id}")
        
        return True
    
    def find_models_by_tag(self, tag: str) -> list:
        """
        Find models with a specific tag
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of model IDs with the tag
        """
        return [
            model_id for model_id, model in self.registry["models"].items()
            if tag in model["tags"]
        ]
    
    def find_models_by_metric(self, metric_name: str, threshold: float, 
                             comparison: str = "greater_than") -> list:
        """
        Find models based on a metric threshold
        
        Args:
            metric_name: Name of the metric to check
            threshold: Threshold value
            comparison: Comparison operator (greater_than, less_than, equal_to)
            
        Returns:
            List of model IDs meeting the criteria
        """
        valid_comparisons = ["greater_than", "less_than", "equal_to"]
        if comparison not in valid_comparisons:
            self.logger.error(f"Invalid comparison: {comparison}. Must be one of {valid_comparisons}")
            return []
        
        matches = []
        
        for model_id, model in self.registry["models"].items():
            if metric_name in model["metrics"]:
                metric_value = model["metrics"][metric_name]
                
                if not isinstance(metric_value, (int, float)):
                    continue
                    
                if comparison == "greater_than" and metric_value > threshold:
                    matches.append(model_id)
                elif comparison == "less_than" and metric_value < threshold:
                    matches.append(model_id)
                elif comparison == "equal_to" and metric_value == threshold:
                    matches.append(model_id)
        
        return matches
    
    def export_model_card(self, model_id: str, output_path: Optional[str] = None) -> str:
        """
        Export a model card with comprehensive information
        
        Args:
            model_id: Model ID to export information for
            output_path: Path to save the model card
            
        Returns:
            Path to the generated model card
        """
        if model_id not in self.registry["models"]:
            self.logger.error(f"Model {model_id} not found in registry")
            return ""
        
        model = self.registry["models"][model_id]
        
        # Create model card content
        model_card = {
            "model_id": model_id,
            "name": model["metadata"].get("name", f"CyberBERT-{model_id[:8]}"),
            "version": model["metadata"].get("version", "1.0.0"),
            "created_at": model["created_at"],
            "status": model["status"],
            "metrics": model["metrics"],
            "metadata": model["metadata"],
            "artifacts": model["artifacts"],
            "deployment_history": model["deployment_history"],
            "tags": model["tags"],
            "timestamp": datetime.now().isoformat(),
            "registry_info": {
                "registry_path": self.registry_dir,
                "model_storage_path": model["registry_path"]
            }
        }
        
        # Determine output path
        if output_path is None:
            output_path = os.path.join(model["registry_path"], "model_card.json")
        
        # Save model card
        with open(output_path, 'w') as f:
            json.dump(model_card, f, indent=2)
        
        self.logger.info(f"Exported model card for {model_id} to {output_path}")
        return output_path
    
    def rollback_model(self, environment: str) -> Optional[str]:
        """
        Rollback the current model in an environment to the previous one
        
        Args:
            environment: Environment to rollback (staging or production)
            
        Returns:
            ID of the model rolled back to, or None if rollback failed
        """
        if environment not in ["staging", "production"]:
            self.logger.error(f"Invalid environment: {environment}")
            return None
        
        env_key = f"{environment}_model"
        current_model_id = self.registry.get(env_key)
        
        if not current_model_id or current_model_id not in self.registry["models"]:
            self.logger.error(f"No current model in {environment} to rollback from")
            return None
        
        # Find previous model for this environment
        current_model = self.registry["models"][current_model_id]
        previous_model_id = None
        
        for deployment in reversed(current_model["deployment_history"]):
            if "superseded_by" in deployment and deployment["superseded_by"] == current_model_id:
                previous_model_id = deployment["model_id"]
                break
        
        if not previous_model_id or previous_model_id not in self.registry["models"]:
            # No previous model found, look for any superseded models
            for model_id, model in self.registry["models"].items():
                if model["status"] == "superseded":
                    for deployment in reversed(model["deployment_history"]):
                        if deployment["environment"] == environment and "superseded_by" in deployment:
                            previous_model_id = model_id
                            break
                    if previous_model_id:
                        break
        
        if not previous_model_id:
            self.logger.error(f"No previous model found for {environment} to rollback to")
            return None
        
        # Update current model status
        current_model["status"] = "rollback_superseded"
        current_model["deployment_history"].append({
            "timestamp": datetime.now().isoformat(),
            "environment": "none",
            "previous_status": environment,
            "rollback_to": previous_model_id
        })
        
        # Promote previous model to environment
        previous_model = self.registry["models"][previous_model_id]
        previous_model["status"] = environment
        previous_model["deployment_history"].append({
            "timestamp": datetime.now().isoformat(),
            "environment": environment,
            "previous_status": previous_model["status"],
            "rollback_from": current_model_id
        })
        
        # Update environment pointer
        self.registry[env_key] = previous_model_id
        self.registry["last_updated"] = datetime.now().isoformat()
        
        # Save registry
        self._save_registry()
        
        self.logger.info(f"Rolled back {environment} from {current_model_id} to {previous_model_id}")
        return previous_model_id