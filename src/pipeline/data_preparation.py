"""
Data preparation module for Gilliam Networks analysis.

Handles loading and validation of prepared child narrative data.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DataPreparation:
    """
    Prepare and validate child narrative data for embedding generation.

    This module handles:
    - Loading prepared text files from embedding_ready directory
    - Validating metadata
    - Checking data integrity
    """

    def __init__(self, config: Dict):
        """
        Initialize data preparation module.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.raw_data_dir = Path(config['paths']['raw_data'])
        self.metadata_path = Path(config['paths']['metadata'])
        self.metadata_mlu_path = Path(config['paths']['metadata_with_mlu'])

    def run(self, force: bool = False) -> Dict:
        """
        Run data preparation and validation.

        Args:
            force: Force re-run even if data already prepared

        Returns:
            Dictionary with preparation statistics
        """
        logger.info("Starting data preparation")

        # Check if data directory exists
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.raw_data_dir}")

        # Load and validate metadata
        metadata = self._load_metadata()

        # Validate text files
        stats = self._validate_text_files(metadata)

        # Generate summary statistics
        summary = self._generate_summary(metadata, stats)

        logger.info(f"Data preparation complete: {stats['valid_files']} files ready")
        return summary

    def _load_metadata(self) -> pd.DataFrame:
        """Load and validate metadata."""
        logger.info(f"Loading metadata from {self.metadata_path}")

        # Try loading MLU metadata first, fall back to regular metadata
        if self.metadata_mlu_path.exists():
            metadata = pd.read_csv(self.metadata_mlu_path)
            logger.info(f"Loaded metadata with MLU: {len(metadata)} records")
        elif self.metadata_path.exists():
            metadata = pd.read_csv(self.metadata_path)
            logger.info(f"Loaded metadata: {len(metadata)} records")
        else:
            raise FileNotFoundError("No metadata file found")

        # Validate required columns
        required_cols = ['filename', 'development_type', 'age_years']
        missing_cols = [col for col in required_cols if col not in metadata.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in metadata: {missing_cols}")

        return metadata

    def _validate_text_files(self, metadata: pd.DataFrame) -> Dict:
        """Validate that all text files exist."""
        stats = {
            'total_files': len(metadata),
            'valid_files': 0,
            'missing_files': [],
            'td_count': 0,
            'sli_count': 0
        }

        for _, row in metadata.iterrows():
            file_path = self.raw_data_dir / row['filename']
            if file_path.exists():
                stats['valid_files'] += 1
                if row['development_type'] == 'TD':
                    stats['td_count'] += 1
                else:
                    stats['sli_count'] += 1
            else:
                stats['missing_files'].append(row['filename'])

        if stats['missing_files']:
            logger.warning(f"Missing {len(stats['missing_files'])} files")
            logger.debug(f"Missing files: {stats['missing_files'][:5]}...")

        return stats

    def _generate_summary(self, metadata: pd.DataFrame, stats: Dict) -> Dict:
        """Generate summary statistics."""
        summary = {
            'total_children': len(metadata),
            'valid_files': stats['valid_files'],
            'td_count': stats['td_count'],
            'sli_count': stats['sli_count'],
            'age_distribution': metadata.groupby('age_years')['development_type'].value_counts().to_dict(),
            'missing_files': len(stats['missing_files'])
        }

        # Add MLU statistics if available
        if 'mlu' in metadata.columns:
            summary['mlu_stats'] = {
                'td_mean': metadata[metadata['development_type'] == 'TD']['mlu'].mean(),
                'td_std': metadata[metadata['development_type'] == 'TD']['mlu'].std(),
                'sli_mean': metadata[metadata['development_type'] == 'SLI']['mlu'].mean(),
                'sli_std': metadata[metadata['development_type'] == 'SLI']['mlu'].std()
            }

        return summary