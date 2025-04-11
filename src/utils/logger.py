import logging
import os
import sys
from datetime import datetime

class Logger:
    """Unified logging system for CyberBERT project"""
    
    LOG_LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    def __init__(self, name="cyberbert", log_level="INFO", log_to_file=True, log_to_console=True):
        """
        Initialize the logger
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Whether to log to file
            log_to_console: Whether to log to console
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.LOG_LEVELS.get(log_level.upper(), logging.INFO))
        self.logger.handlers = []  # Clear existing handlers
        
        # Create formatter
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_to_file:
            # Create logs directory if it doesn't exist
            logs_dir = os.path.join(os.getcwd(), 'logs')
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)
            
            # Create log file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(logs_dir, f"{name}_{timestamp}.log")
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def get_logger(self):
        """Get the configured logger"""
        return self.logger