#!/usr/bin/env python
"""
Main pipeline runner for Gilliam Networks analysis.
Run complete pipeline or individual phases.

Usage:
    python scripts/run_pipeline.py --phase all
    python scripts/run_pipeline.py --phase embed --config custom_config.yaml
    python scripts/run_pipeline.py --phase analyze --verbose
"""

import click
import yaml
import sys
import logging
from pathlib import Path
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import pipeline modules (we'll create these next)
from src.pipeline.data_preparation import DataPreparation
from src.pipeline.embedding_generation import EmbeddingGenerator
from src.pipeline.network_construction import NetworkConstructor
from src.pipeline.analysis import NetworkAnalyzer
from src.utils.logging_utils import setup_logging
from src.utils.progress import ProgressTracker


@click.command()
@click.option('--config', default='config.yaml', help='Configuration file path')
@click.option('--phase',
              type=click.Choice(['all', 'prepare', 'embed', 'network', 'analyze']),
              default='all',
              help='Which phase to run')
@click.option('--verbose', is_flag=True, help='Enable verbose output')
@click.option('--dry-run', is_flag=True, help='Show what would be done without executing')
@click.option('--force', is_flag=True, help='Force re-run even if outputs exist')
def main(config, phase, verbose, dry_run, force):
    """
    Gilliam Networks Analysis Pipeline

    Analyze semantic networks in child language development using BERT embeddings
    to distinguish between typically developing (TD) and specific language
    impairment (SLI) children.
    """

    # Load configuration
    try:
        with open(config) as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        click.echo(f"❌ Configuration file not found: {config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        click.echo(f"❌ Error parsing configuration: {e}")
        sys.exit(1)

    # Setup logging
    setup_logging(verbose=verbose, log_file=cfg.get('logging', {}).get('log_file', 'pipeline.log'))
    logger = logging.getLogger(__name__)

    # Initialize progress tracker
    progress = ProgressTracker()

    # Pipeline header
    click.echo("=" * 60)
    click.echo("🧠 Gilliam Networks Analysis Pipeline")
    click.echo("=" * 60)
    click.echo(f"Configuration: {config}")
    click.echo(f"Phase: {phase}")
    click.echo(f"Verbose: {verbose}")
    click.echo(f"Dry run: {dry_run}")
    click.echo("=" * 60)

    if dry_run:
        click.echo("\n📋 DRY RUN MODE - No actual processing will occur")

    start_time = time.time()

    try:
        # Phase 1: Data Preparation
        if phase in ['all', 'prepare']:
            click.echo("\n" + "=" * 60)
            click.echo("📝 Phase 1: Data Preparation")
            click.echo("=" * 60)

            if dry_run:
                click.echo("Would prepare data from embedding_ready directory")
                click.echo("Would generate metadata and combined text files")
            else:
                logger.info("Starting data preparation phase")
                data_prep = DataPreparation(cfg)
                data_prep.run(force=force)
                progress.mark_complete('data_prep')
                click.echo("✅ Data preparation complete")

        # Phase 2: Embedding Generation
        if phase in ['all', 'embed']:
            click.echo("\n" + "=" * 60)
            click.echo("🤖 Phase 2: Embedding Generation")
            click.echo("=" * 60)

            if dry_run:
                click.echo(f"Would generate embeddings using {cfg['embeddings']['model']}")
                click.echo(f"Device: {cfg['embeddings']['device']}")
                click.echo(f"Batch size: {cfg['embeddings']['batch_size']}")
            else:
                logger.info("Starting embedding generation phase")
                generator = EmbeddingGenerator(cfg)
                generator.run(
                    input_dir=cfg['paths']['raw_data'],
                    output_dir=cfg['paths']['embeddings'],
                    force=force
                )
                progress.mark_complete('embeddings')
                click.echo("✅ Embedding generation complete")

        # Phase 3: Network Construction
        if phase in ['all', 'network']:
            click.echo("\n" + "=" * 60)
            click.echo("🕸️  Phase 3: Network Construction")
            click.echo("=" * 60)

            if dry_run:
                click.echo(f"Would construct networks using {cfg['network']['construction_method']} method")
                click.echo(f"Similarity threshold: {cfg['network']['similarity_threshold']}")
            else:
                logger.info("Starting network construction phase")
                constructor = NetworkConstructor(cfg)
                constructor.run(
                    embeddings_dir=cfg['paths']['embeddings'],
                    output_dir=cfg['paths']['networks'],
                    force=force
                )
                progress.mark_complete('networks')
                click.echo("✅ Network construction complete")

        # Phase 4: Analysis
        if phase in ['all', 'analyze']:
            click.echo("\n" + "=" * 60)
            click.echo("📊 Phase 4: Analysis & Visualization")
            click.echo("=" * 60)

            if dry_run:
                click.echo("Would analyze network properties and generate visualizations")
                click.echo(f"Bootstrap iterations: {cfg['analysis']['bootstrap_iterations']}")
                click.echo(f"Age groups: {cfg['analysis']['age_groups']}")
            else:
                logger.info("Starting analysis phase")
                analyzer = NetworkAnalyzer(cfg)
                results = analyzer.run(
                    networks_dir=cfg['paths']['networks'],
                    output_dir=cfg['paths']['results'],
                    force=force
                )
                progress.mark_complete('analysis')

                # Generate summary report
                click.echo("\n📈 Analysis Summary:")
                if results:
                    click.echo(f"  - TD children analyzed: {results.get('td_count', 0)}")
                    click.echo(f"  - SLI children analyzed: {results.get('sli_count', 0)}")
                    click.echo(f"  - Optimal threshold: {results.get('optimal_threshold', 0):.3f}")
                    click.echo(f"  - Max small-world σ: {results.get('max_sigma', 0):.2f}")
                click.echo("✅ Analysis complete")

    except KeyboardInterrupt:
        click.echo("\n\n⚠️  Pipeline interrupted by user")
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n\n❌ Pipeline failed: {e}")
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

    # Pipeline complete
    elapsed_time = time.time() - start_time
    elapsed_str = f"{elapsed_time/60:.1f} minutes" if elapsed_time > 60 else f"{elapsed_time:.1f} seconds"

    click.echo("\n" + "=" * 60)
    click.echo("✨ Pipeline Complete!")
    click.echo("=" * 60)
    click.echo(f"Total time: {elapsed_str}")
    click.echo(f"Results saved to: {cfg['paths']['results']}")

    # Show progress summary
    click.echo("\nProgress Summary:")
    for phase_name, completed in progress.phases.items():
        status = "✅" if completed else "⏭️"
        click.echo(f"  {status} {phase_name.replace('_', ' ').title()}")

    if not dry_run:
        click.echo(f"\nView results in: {cfg['paths']['results']}")
        click.echo("Run 'make explore' to launch interactive exploration")


if __name__ == '__main__':
    main()