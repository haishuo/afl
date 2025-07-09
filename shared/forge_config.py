"""
AFL Forge Configuration
=======================

Configuration utilities for AFL project on Forge infrastructure.
Provides standardized path management for AFL data, code, and artifacts.
"""

from pathlib import Path
from typing import Optional


class AFLConfig:
    """Configuration class for AFL project paths and settings."""
    
    def __init__(self):
        # Base Forge directories
        self.base_dir = Path("/mnt")
        
        # AFL-specific top-level directories
        self.data_dir = self.base_dir / "data" / "afl"
        self.projects_dir = self.base_dir / "projects" / "afl"
        self.artifacts_dir = self.base_dir / "artifacts" / "afl"
        
        # Data subdirectories
        self.datasets_dir = self.data_dir / "datasets"
        self.experiments_dir = self.data_dir / "experiments"
        self.common_datasets_dir = self.base_dir / "data" / "common_datasets"
        
        # Project code subdirectories
        self.core_dir = self.projects_dir / "core"
        self.scripts_dir = self.projects_dir / "scripts"
        self.shared_dir = self.projects_dir / "shared"
        self.tests_dir = self.projects_dir / "tests"
        
        # Artifacts subdirectories
        self.tables_dir = self.artifacts_dir / "tables"
        self.master_tables_dir = self.tables_dir / "master"
        self.individual_tables_dir = self.tables_dir / "individual"
        self.formatted_tables_dir = self.tables_dir / "formatted"
        
        self.reports_dir = self.artifacts_dir / "reports"
        self.experiment_reports_dir = self.reports_dir / "experiment_reports"
        self.statistical_analysis_dir = self.reports_dir / "statistical_analysis"
        self.methodology_dir = self.reports_dir / "methodology"
        
        self.papers_dir = self.artifacts_dir / "papers"
        self.drafts_dir = self.papers_dir / "drafts"
        self.final_papers_dir = self.papers_dir / "final"
        self.supplementary_dir = self.papers_dir / "supplementary"
        
        self.visualizations_dir = self.artifacts_dir / "visualizations"
        self.plots_dir = self.visualizations_dir / "plots"
        self.figures_dir = self.visualizations_dir / "figures"
        self.interactive_dir = self.visualizations_dir / "interactive"
        
        self.raw_results_dir = self.artifacts_dir / "raw_results"
        self.json_results_dir = self.raw_results_dir / "json"
        self.csv_results_dir = self.raw_results_dir / "csv"
        self.logs_dir = self.raw_results_dir / "logs"
    
    def get_experiment_dir(self, experiment_name: str) -> Path:
        """Get the data directory for a specific experiment."""
        return self.experiments_dir / experiment_name
    
    def get_experiment_models_dir(self, experiment_name: str) -> Path:
        """Get the models directory for a specific experiment."""
        return self.get_experiment_dir(experiment_name) / "models"
    
    def get_experiment_results_dir(self, experiment_name: str) -> Path:
        """Get the results directory for a specific experiment."""
        return self.get_experiment_dir(experiment_name) / "results"
    
    def get_experiment_config_path(self, experiment_name: str) -> Path:
        """Get the config file path for a specific experiment."""
        return self.get_experiment_dir(experiment_name) / "config.yaml"
    
    def get_dataset_path(self, dataset_name: str) -> Path:
        """Get the path to a specific dataset (via symlink)."""
        return self.datasets_dir / dataset_name
    
    def get_common_dataset_path(self, dataset_name: str) -> Path:
        """Get the path to a dataset in common_datasets (original location)."""
        return self.common_datasets_dir / dataset_name
    
    def get_master_table_path(self, format_type: str = "json") -> Path:
        """Get the path to the master AFL table."""
        filename = f"afl_master_table.{format_type}"
        return self.master_tables_dir / filename
    
    def get_experiment_report_path(self, experiment_name: str) -> Path:
        """Get the path for an individual experiment report."""
        filename = f"{experiment_name}_report.json"
        return self.experiment_reports_dir / filename
    
    def get_raw_results_path(self, experiment_name: str, format_type: str = "json") -> Path:
        """Get the path for raw experiment results."""
        filename = f"{experiment_name}_raw_results.{format_type}"
        if format_type == "json":
            return self.json_results_dir / filename
        elif format_type == "csv":
            return self.csv_results_dir / filename
        else:
            return self.raw_results_dir / filename
    
    def get_log_path(self, experiment_name: str) -> Path:
        """Get the log file path for an experiment."""
        filename = f"{experiment_name}.log"
        return self.logs_dir / filename
    
    def ensure_directories(self) -> None:
        """Create all necessary directories if they don't exist."""
        # Core directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Data subdirectories
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # Artifacts subdirectories
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.master_tables_dir.mkdir(parents=True, exist_ok=True)
        self.individual_tables_dir.mkdir(parents=True, exist_ok=True)
        self.formatted_tables_dir.mkdir(parents=True, exist_ok=True)
        
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_reports_dir.mkdir(parents=True, exist_ok=True)
        self.statistical_analysis_dir.mkdir(parents=True, exist_ok=True)
        self.methodology_dir.mkdir(parents=True, exist_ok=True)
        
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.drafts_dir.mkdir(parents=True, exist_ok=True)
        self.final_papers_dir.mkdir(parents=True, exist_ok=True)
        self.supplementary_dir.mkdir(parents=True, exist_ok=True)
        
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.interactive_dir.mkdir(parents=True, exist_ok=True)
        
        self.raw_results_dir.mkdir(parents=True, exist_ok=True)
        self.json_results_dir.mkdir(parents=True, exist_ok=True)
        self.csv_results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def __str__(self) -> str:
        """String representation of AFL configuration."""
        return f"""AFL Configuration:
  Data Directory: {self.data_dir}
  Projects Directory: {self.projects_dir}
  Artifacts Directory: {self.artifacts_dir}
  
  Key Paths:
  - Datasets: {self.datasets_dir}
  - Experiments: {self.experiments_dir}
  - Master Tables: {self.master_tables_dir}
  - Reports: {self.reports_dir}
  - Raw Results: {self.raw_results_dir}
"""


# Global configuration instance
_afl_config: Optional[AFLConfig] = None


def get_afl_config() -> AFLConfig:
    """Get the global AFL configuration instance."""
    global _afl_config
    if _afl_config is None:
        _afl_config = AFLConfig()
    return _afl_config


def initialize_afl_directories() -> None:
    """Initialize all AFL directories."""
    config = get_afl_config()
    config.ensure_directories()


# Convenience functions for common paths
def get_datasets_dir() -> Path:
    """Get the AFL datasets directory."""
    return get_afl_config().datasets_dir


def get_experiments_dir() -> Path:
    """Get the AFL experiments directory."""
    return get_afl_config().experiments_dir


def get_artifacts_dir() -> Path:
    """Get the AFL artifacts directory."""
    return get_afl_config().artifacts_dir


def get_master_tables_dir() -> Path:
    """Get the AFL master tables directory."""
    return get_afl_config().master_tables_dir


if __name__ == "__main__":
    # Test the configuration
    config = get_afl_config()
    print(config)
    
    # Initialize directories
    print("Initializing AFL directories...")
    initialize_afl_directories()
    print("✅ AFL directories initialized")