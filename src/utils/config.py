import os
import yaml
import json
import platform
from typing import Dict, Any, Optional

class Config:
    """Simplified configuration management system for CyberBERT"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration, loading from file if provided
        
        Args:
            config_path: Path to config file (.yaml or .json)
        """
        # Default configuration with essential settings
        self.config = {
            # System settings
            'system': {
                'log_level': 'INFO',
                'log_to_file': True,
            },
            
            # Data settings
            'data': {
                'data_path': 'data/processed/clean_data.csv',
                'feature_selection': True,
                'feature_count': 30,
                'sample_fraction': 1.0,
                'max_length': 256,
                'cache_tokenization': True,
            },
            
            # Model settings
            'model': {
                'model_path': 'models/cyberbert_model',
                'output_dir': 'models/trained_cyberbert',
                'learning_rate': 2e-5,
                'epochs': 5,
                'batch_size': 16,
                'mixed_precision': True,
                'early_stopping': 3,
                'weight_decay': 0.01,
                'warmup_ratio': 0.1,
                'eval_steps': 100,
                'save_steps': 500,
                'labels': [
                    "BENIGN",
                    "DDoS",
                    "PortScan", 
                    "FTP-Patator",
                    "SSH-Patator",
                    "DoS slowloris",
                    "DoS Slowhttptest",
                    "DoS GoldenEye"
                ]
            }
        }
        
        # Load configuration from file if provided
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from file
        
        Args:
            config_path: Path to config file (.yaml or .json)
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        file_ext = os.path.splitext(config_path)[1].lower()
        
        try:
            if file_ext == '.yaml' or file_ext == '.yml':
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
            elif file_ext == '.json':
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {file_ext}")
            
            # Update configuration with loaded values
            self._update_nested_dict(self.config, loaded_config)
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {str(e)}")
    
    def save_config(self, config_path: str) -> None:
        """
        Save configuration to file
        
        Args:
            config_path: Path to save config file (.yaml or .json)
        """
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            file_ext = os.path.splitext(config_path)[1].lower()
            
            if file_ext == '.yaml' or file_ext == '.yml':
                with open(config_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            elif file_ext == '.json':
                with open(config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {file_ext}")
        except Exception as e:
            raise ValueError(f"Failed to save configuration: {str(e)}")
    
    def update_from_args(self, args) -> None:
        """
        Update configuration from command line arguments
        
        Args:
            args: Parsed command line arguments
        """
        for arg_name, arg_value in vars(args).items():
            if arg_value is not None:
                self._update_config_from_arg(arg_name, arg_value)
    
    def _update_config_from_arg(self, arg_name: str, arg_value: Any) -> None:
        """Map a command line arg to the correct config location"""
        # Data settings
        if arg_name == 'data':
            self.config['data']['data_path'] = arg_value
        elif arg_name == 'feature_count':
            self.config['data']['feature_count'] = arg_value
        elif arg_name == 'sample_frac':
            self.config['data']['sample_fraction'] = arg_value
        elif arg_name == 'max_length':
            self.config['data']['max_length'] = arg_value
        elif arg_name == 'cache_tokenization':
            self.config['data']['cache_tokenization'] = arg_value
        elif arg_name == 'no_feature_selection':
            self.config['data']['feature_selection'] = not arg_value
        
        # Model settings
        elif arg_name == 'model':
            self.config['model']['model_path'] = arg_value
        elif arg_name == 'output':
            self.config['model']['output_dir'] = arg_value
        elif arg_name == 'learning_rate':
            self.config['model']['learning_rate'] = arg_value
        elif arg_name == 'epochs':
            self.config['model']['epochs'] = arg_value
        elif arg_name == 'batch_size':
            self.config['model']['batch_size'] = arg_value
        elif arg_name == 'mixed_precision':
            self.config['model']['mixed_precision'] = arg_value
        elif arg_name == 'early_stopping':
            self.config['model']['early_stopping'] = arg_value
        elif arg_name == 'eval_steps':
            self.config['model']['eval_steps'] = arg_value
        
        # System settings
        elif arg_name == 'log_level':
            self.config['system']['log_level'] = arg_value
        elif arg_name == 'no_log_file':
            self.config['system']['log_to_file'] = not arg_value
    
    def _update_nested_dict(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a nested dictionary with values from another dictionary
        
        Args:
            d: Target dictionary to update
            u: Source dictionary with new values
        
        Returns:
            Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the configuration
        
        Args:
            key: Dot-separated path to the value (e.g., 'model.learning_rate')
            default: Default value if key not found
        
        Returns:
            Value from configuration or default
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default