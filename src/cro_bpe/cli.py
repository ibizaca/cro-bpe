import argparse
import logging

from cro_bpe.config import CorpusConfig, PipelineConfig, TokenizerConfig
from cro_bpe.logging_utils import configure_logging
from cro_bpe.training import prepare_corpus, run_training_pipeline, train_tokenizer
from cro_bpe.wiki import download_croatian_wikipedia_dump

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Croatian-only BPE tokenizer.")
    subparsers = parser.add_subparsers(dest="command", required=False)

    for command_name in ["download", "prepare-corpus", "train"]:
        command_parser = subparsers.add_parser(command_name)
        command_parser.add_argument("--article-limit", type=int, default=None)
        command_parser.add_argument("--min-document-characters", type=int, default=200)
        command_parser.add_argument("--vocab-size", type=int, default=32000)
        command_parser.add_argument("--min-frequency", type=int, default=2)

    parser.set_defaults(command="train")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        corpus=CorpusConfig(
            article_limit=args.article_limit,
            min_document_characters=args.min_document_characters,
        ),
        tokenizer=TokenizerConfig(
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        ),
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    configure_logging(config.directories.logs_directory)

    logger.info("Using Croatian-only source: %s", config.source.dump_url)

    if args.command == "download":
        download_croatian_wikipedia_dump(config)
        return

    if args.command == "prepare-corpus":
        prepare_corpus(config)
        return

    if args.command == "train":
        run_training_pipeline(config)
        return

    train_tokenizer(config)
