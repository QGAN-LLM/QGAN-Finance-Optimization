"""
Logging setup for the research project.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
import datetime

def setup_logging(log_dir: Optional[Path] = None,
                 log_level: int = logging.INFO,
                 console_level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: File logging level
        console_level: Console logging level
        
    Returns:
        Root logger
    """
    # Create log directory if specified
    if log_dir:
        log_dir.mkdir(exist_ok=True, parents=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"research_{timestamp}.log"
    else:
        log_file = None
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(min(log_level, console_level))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if log directory specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        root_logger.info(f"Log file: {log_file}")
    
    # Add custom handlers for specific modules
    _setup_module_loggers()
    
    root_logger.info("Logging setup complete")
    return root_logger

def _setup_module_loggers():
    """Configure loggers for specific modules."""
    # Quantum module logging
    quantum_logger = logging.getLogger('src.quantum')
    quantum_logger.setLevel(logging.DEBUG)
    
    # Data pipeline logging
    data_logger = logging.getLogger('src.data_pipeline')
    data_logger.setLevel(logging.DEBUG)
    
    # Evaluation logging
    eval_logger = logging.getLogger('src.evaluation')
    eval_logger.setLevel(logging.DEBUG)

def get_logger(name: str) -> logging.Logger:
    """
    Get logger for a specific module.
    
    Args:
        name: Module name (e.g., 'src.quantum.qgan')
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)