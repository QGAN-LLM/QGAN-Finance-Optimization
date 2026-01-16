#!/usr/bin/env python
"""Main script to run the full QGAN-LLM experiment pipeline."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import logging
from datetime import datetime

from src.utils.config import ExperimentConfig, ConfigManager
from src.utils.experiment_tracker import ExperimentTracker
from src.data_pipeline.acquisition import DataAcquirer
from src.data_pipeline.cleaning import DataCleaner
from src.data_pipeline.feature_engineering import FeatureEngineer
from src.quantum.qgan import QGANModel
from src.classical.discriminator import ClassicalDiscriminator
from src.evaluation.metrics import FINSECBenchmark
from src.evaluation.attacks import AdversarialAttacker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run QGAN-LLM cybersecurity experiment")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data-only", action="store_true",
                       help="Run only data pipeline")
    parser.add_argument("--train-only", action="store_true",
                       help="Run only model training")
    parser.add_argument("--eval-only", action="store_true",
                       help="Run only evaluation")
    parser.add_argument("--experiment-name", type=str, default="qgan_llm_experiment",
                       help="Name for experiment tracking")
    return parser.parse_args()

def run_data_pipeline(config: ExperimentConfig, tracker: ExperimentTracker):
    """Run the complete data pipeline."""
    logger.info("Starting data pipeline...")
    
    # 1. Data Acquisition
    acquirer = DataAcquirer(config.data)
    raw_data = acquirer.download_data()
    tracker.log_artifact(acquirer.metadata_path, "data_metadata")
    
    # 2. Data Cleaning
    cleaner = DataCleaner(config.data)
    cleaned_data = cleaner.clean(raw_data)
    logger.info(f"Cleaned data shape: {cleaned_data.shape}")
    
    # 3. Feature Engineering
    engineer = FeatureEngineer(config.data)
    processed_data, features = engineer.transform(cleaned_data)
    
    # Save processed data
    processed_path = config.data_dir / "processed" / "dataset.feather"
    processed_data.to_feather(processed_path)
    tracker.log_artifact(processed_path, "processed_data")
    
    return processed_data, features

def run_qgan_training(config: ExperimentConfig, data, tracker: ExperimentTracker):
    """Train the QGAN model."""
    logger.info("Starting QGAN training...")
    
    # Initialize QGAN
    qgan = QGANModel(
        quantum_config=config.quantum,
        classical_config=config.classical
    )
    
    # Split data
    train_data, val_data, test_data = qgan.split_data(data)
    
    # Train
    training_history = qgan.train(
        train_data=train_data,
        val_data=val_data,
        epochs=config.quantum.n_epochs,
        tracker=tracker
    )
    
    # Generate synthetic data
    synthetic_data = qgan.generate_synthetic(n_samples=len(train_data))
    
    # Save model and synthetic data
    model_path = config.models_dir / f"qgan_model_{tracker.experiment_id}.pth"
    qgan.save(model_path)
    tracker.log_artifact(model_path, "qgan_model")
    
    synthetic_path = config.data_dir / "synthetic" / f"synthetic_data_{tracker.experiment_id}.feather"
    synthetic_data.to_feather(synthetic_path)
    tracker.log_artifact(synthetic_path, "synthetic_data")
    
    return qgan, synthetic_data

def run_evaluation(config: ExperimentConfig, qgan, real_data, synthetic_data, tracker: ExperimentTracker):
    """Run comprehensive evaluation."""
    logger.info("Starting evaluation pipeline...")
    
    # Initialize benchmark
    benchmark = FINSECBenchmark(config.evaluation)
    
    # 1. Forecast Accuracy Evaluation
    logger.info("Evaluating forecast accuracy...")
    forecast_metrics = benchmark.evaluate_forecast_accuracy(
        real_data=real_data,
        synthetic_data=synthetic_data,
        model=qgan
    )
    tracker.log_metrics(forecast_metrics, step=0)
    
    # 2. Adversarial Robustness Evaluation
    logger.info("Evaluating adversarial robustness...")
    attacker = AdversarialAttacker(config.attack)
    
    attack_results = {}
    for attack_type in config.attack.attack_types:
        logger.info(f"Running {attack_type.upper()} attack...")
        results = attacker.attack_model(
            model=qgan,
            data=real_data,
            attack_type=attack_type
        )
        attack_results[attack_type] = results
        tracker.log_metrics({f"asr_{attack_type}": results["success_rate"]}, step=0)
    
    # 3. Quantum State Analysis (for RQ5)
    logger.info("Performing quantum state tomography...")
    quantum_state = qgan.get_quantum_state()
    tomography_results = benchmark.analyze_quantum_state(quantum_state)
    tracker.log_quantum_state(quantum_state, "final_state")
    tracker.log_metrics(tomography_results, step=0)
    
    # 4. Statistical Significance Testing
    logger.info("Running statistical tests...")
    stats_results = benchmark.run_statistical_tests(
        real_data=real_data,
        synthetic_data=synthetic_data,
        attack_results=attack_results
    )
    tracker.log_metrics(stats_results, step=0)
    
    # 5. Generate comprehensive report
    report = benchmark.generate_report(
        forecast_metrics=forecast_metrics,
        attack_results=attack_results,
        quantum_analysis=tomography_results,
        stats_results=stats_results
    )
    
    report_path = config.results_dir / f"final_report_{tracker.experiment_id}.md"
    with open(report_path, 'w') as f:
        f.write(report)
    tracker.log_artifact(report_path, "final_report")
    
    return report

def main():
    """Main execution function."""
    args = parse_args()
    
    # Load configuration
    config_manager = ConfigManager(Path(args.config))
    config = config_manager.load_config()
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(
        experiment_name=args.experiment_name,
        config=config.__dict__,
        use_mlflow=True,
        use_wandb=True
    )
    
    try:
        # Run complete pipeline or specific stages
        if args.data_only or not (args.train_only or args.eval_only):
            data, features = run_data_pipeline(config, tracker)
        
        if args.train_only or not (args.data_only or args.eval_only):
            qgan, synthetic_data = run_qgan_training(config, data, tracker)
        
        if args.eval_only or not (args.data_only or args.train_only):
            # In eval-only mode, load existing models/data
            if args.eval_only:
                # Load pre-trained model and data
                # (Implementation depends on your storage strategy)
                pass
            report = run_evaluation(config, qgan, data, synthetic_data, tracker)
            logger.info(f"Evaluation complete. Report saved.")
            
            # Print key findings
            print("\n" + "="*50)
            print("EXPERIMENT COMPLETE")
            print("="*50)
            print(f"Experiment ID: {tracker.experiment_id}")
            print(f"Report saved to: results/final_report_{tracker.experiment_id}.md")
            print("="*50)
    
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}", exc_info=True)
        raise
    finally:
        tracker.close()
        logger.info("Experiment tracking closed.")

if __name__ == "__main__":
    main()