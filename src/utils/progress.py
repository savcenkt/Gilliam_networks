"""
Progress tracking utility for the Gilliam Networks pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Track pipeline progress across phases.

    Maintains state of completed phases and can resume from checkpoints.
    """

    def __init__(self, checkpoint_file: Optional[str] = "pipeline_progress.json"):
        """
        Initialize progress tracker.

        Args:
            checkpoint_file: Path to checkpoint file for saving progress
        """
        self.checkpoint_file = Path(checkpoint_file) if checkpoint_file else None
        self.phases = {
            'data_prep': False,
            'embeddings': False,
            'networks': False,
            'analysis': False
        }
        self.timestamps = {}
        self.statistics = {}

        # Load existing progress if available
        if self.checkpoint_file and self.checkpoint_file.exists():
            self.load_state()

    def mark_complete(self, phase: str, stats: Optional[Dict] = None):
        """
        Mark a phase as complete.

        Args:
            phase: Name of the phase
            stats: Optional statistics for the phase
        """
        if phase in self.phases:
            self.phases[phase] = True
            self.timestamps[phase] = datetime.now().isoformat()

            if stats:
                self.statistics[phase] = stats

            logger.info(f"Phase '{phase}' marked as complete")
            self.save_state()
        else:
            logger.warning(f"Unknown phase: {phase}")

    def is_complete(self, phase: str) -> bool:
        """
        Check if a phase is complete.

        Args:
            phase: Name of the phase

        Returns:
            True if phase is complete, False otherwise
        """
        return self.phases.get(phase, False)

    def get_incomplete_phases(self) -> list:
        """
        Get list of incomplete phases.

        Returns:
            List of phase names that are not complete
        """
        return [phase for phase, completed in self.phases.items() if not completed]

    def get_progress_percentage(self) -> float:
        """
        Get overall progress percentage.

        Returns:
            Percentage of completed phases (0-100)
        """
        completed = sum(1 for completed in self.phases.values() if completed)
        total = len(self.phases)
        return (completed / total) * 100 if total > 0 else 0

    def save_state(self):
        """Save current progress state to checkpoint file."""
        if self.checkpoint_file:
            try:
                state = {
                    'phases': self.phases,
                    'timestamps': self.timestamps,
                    'statistics': self.statistics,
                    'last_updated': datetime.now().isoformat()
                }

                self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.checkpoint_file, 'w') as f:
                    json.dump(state, f, indent=2)

                logger.debug(f"Progress saved to {self.checkpoint_file}")
            except Exception as e:
                logger.error(f"Failed to save progress: {e}")

    def load_state(self):
        """Load progress state from checkpoint file."""
        if self.checkpoint_file and self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    state = json.load(f)

                self.phases = state.get('phases', self.phases)
                self.timestamps = state.get('timestamps', {})
                self.statistics = state.get('statistics', {})

                logger.info(f"Progress loaded from {self.checkpoint_file}")
                logger.info(f"Progress: {self.get_progress_percentage():.0f}% complete")
            except Exception as e:
                logger.error(f"Failed to load progress: {e}")

    def reset(self):
        """Reset all progress."""
        self.phases = {phase: False for phase in self.phases}
        self.timestamps = {}
        self.statistics = {}

        if self.checkpoint_file and self.checkpoint_file.exists():
            self.checkpoint_file.unlink()

        logger.info("Progress reset")

    def get_summary(self) -> Dict:
        """
        Get progress summary.

        Returns:
            Dictionary with progress information
        """
        return {
            'phases': self.phases,
            'timestamps': self.timestamps,
            'statistics': self.statistics,
            'progress_percentage': self.get_progress_percentage(),
            'incomplete_phases': self.get_incomplete_phases()
        }