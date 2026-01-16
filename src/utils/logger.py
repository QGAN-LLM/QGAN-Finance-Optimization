"""
Logging Utility
Centralized logging for the project
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logger(name: str, 
                log_dir: str = "../logs",
                level: int = logging.INFO,
                console: bool = True,
                file: bool = True) -> logging.Logger:
    """
    Setup logger with console and file handlers
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console: Whether to log to console
        file: Whether to log to file
    
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class EthicalLogger:
    """Specialized logger for ethical compliance tracking"""
    
    def __init__(self, log_dir: str = "../logs/ethical"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Main ethical log file
        self.ethical_log_file = self.log_dir / "ethical_compliance.log"
        
        # Setup ethical logger
        self.logger = logging.getLogger('ethical_compliance')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        # File handler for ethical logs
        file_handler = logging.FileHandler(self.ethical_log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Also log to console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def log_data_minimization(self, action: str, data_type: str, size: int):
        """Log data minimization actions"""
        message = f"DATA_MINIMIZATION - {action} - Type: {data_type}, Size: {size}"
        self.logger.info(message)
    
    def log_synthetic_data_generation(self, batch_size: int, compliance: bool, 
                                     violations: Optional[list] = None):
        """Log synthetic data generation with compliance check"""
        status = "COMPLIANT" if compliance else "NON-COMPLIANT"
        message = f"SYNTHETIC_DATA - Batch: {batch_size}, Status: {status}"
        
        if not compliance and violations:
            message += f", Violations: {violations}"
        
        if compliance:
            self.logger.info(message)
        else:
            self.logger.warning(message)
    
    def log_purpose_limitation(self, purpose: str, data_usage: str, 
                              permitted: bool):
        """Log purpose limitation checks"""
        status = "PERMITTED" if permitted else "PROHIBITED"
        message = f"PURPOSE_LIMITATION - Purpose: {purpose}, Usage: {data_usage}, Status: {status}"
        
        if permitted:
            self.logger.info(message)
        else:
            self.logger.error(message)
    
    def log_ethical_violation(self, violation_type: str, description: str, 
                             severity: str = "MEDIUM"):
        """Log ethical violations"""
        message = f"ETHICAL_VIOLATION - Type: {violation_type}, Severity: {severity}, Desc: {description}"
        self.logger.error(message)
    
    def generate_compliance_report(self) -> dict:
        """Generate compliance report from logs"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'data_minimization_events': 0,
            'synthetic_data_events': 0,
            'purpose_limitation_checks': 0,
            'ethical_violations': 0,
            'compliance_rate': 1.0
        }
        
        # Read log file and count events
        if self.ethical_log_file.exists():
            with open(self.ethical_log_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                if "DATA_MINIMIZATION" in line:
                    report['data_minimization_events'] += 1
                elif "SYNTHETIC_DATA" in line:
                    report['synthetic_data_events'] += 1
                elif "PURPOSE_LIMITATION" in line:
                    report['purpose_limitation_checks'] += 1
                elif "ETHICAL_VIOLATION" in line:
                    report['ethical_violations'] += 1
            
            # Calculate compliance rate
            total_events = (report['data_minimization_events'] + 
                          report['synthetic_data_events'] + 
                          report['purpose_limitation_checks'])
            
            if total_events > 0:
                report['compliance_rate'] = 1 - (report['ethical_violations'] / total_events)
        
        return report


class TrainingLogger:
    """Specialized logger for training progress"""
    
    def __init__(self, experiment_name: str, log_dir: str = "../logs/training"):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup training logger
        self.logger = logging.getLogger(f'training_{experiment_name}')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        # Training log file
        log_file = self.log_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - Epoch %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Metrics file (CSV)
        self.metrics_file = self.log_dir / "metrics.csv"
        self._initialize_metrics_file()
    
    def _initialize_metrics_file(self):
        """Initialize metrics CSV file with headers"""
        with open(self.metrics_file, 'w') as f:
            f.write("epoch,train_loss,val_loss,ethical_score,learning_rate,epoch_time\n")
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                 ethical_score: float, learning_rate: float, epoch_time: float):
        """Log epoch metrics"""
        # Log to file
        message = f"{epoch:4d} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, Ethical: {ethical_score:.3f}"
        self.logger.info(message)
        
        # Write to CSV
        with open(self.metrics_file, 'a') as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{ethical_score:.6f},{learning_rate:.6f},{epoch_time:.2f}\n")
    
    def log_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Log checkpoint creation"""
        status = "BEST" if is_best else "REGULAR"
        message = f"Checkpoint saved - Epoch: {epoch}, Loss: {loss:.4f}, Status: {status}"
        self.logger.info(message)
    
    def log_early_stopping(self, epoch: int, best_epoch: int, best_loss: float):
        """Log early stopping event"""
        message = f"Early stopping - Stopped at: {epoch}, Best epoch: {best_epoch}, Best loss: {best_loss:.4f}"
        self.logger.info(message)
    
    def get_training_summary(self) -> dict:
        """Get training summary from logs"""
        summary = {
            'experiment_name': self.experiment_name,
            'log_file': str(self.log_dir / "training.log"),
            'metrics_file': str(self.metrics_file)
        }
        
        # Read metrics file
        if self.metrics_file.exists():
            try:
                import pandas as pd
                metrics_df = pd.read_csv(self.metrics_file)
                
                if not metrics_df.empty:
                    summary['total_epochs'] = len(metrics_df)
                    summary['final_train_loss'] = float(metrics_df['train_loss'].iloc[-1])
                    summary['final_val_loss'] = float(metrics_df['val_loss'].iloc[-1])
                    summary['best_val_loss'] = float(metrics_df['val_loss'].min())
                    summary['best_epoch'] = int(metrics_df['val_loss'].idxmin() + 1)
                    summary['average_ethical_score'] = float(metrics_df['ethical_score'].mean())
                    summary['total_training_time'] = float(metrics_df['epoch_time'].sum())
            except Exception as e:
                summary['error'] = str(e)
        
        return summary