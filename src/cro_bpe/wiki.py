import bz2
import logging
import shutil
import urllib.request
import xml.etree.ElementTree as element_tree
from pathlib import Path

from cro_bpe.config import PipelineConfig
from cro_bpe.normalize import normalize_croatian_text

logger = logging.getLogger(__name__)


def ensure_directories(config: PipelineConfig) -> None:
    config.directories.raw_directory.mkdir(parents=True, exist_ok=True)
    config.directories.preprocessed_directory.mkdir(parents=True, exist_ok=True)
    config.directories.models_directory.mkdir(parents=True, exist_ok=True)


def download_croatian_wikipedia_dump(config: PipelineConfig) -> Path:
    ensure_directories(config)
    if config.dump_path.exists() and config.dump_path.stat().st_size > 0:
        logger.info("Reusing existing dump at %s", config.dump_path)
        return config.dump_path

    temporary_path = config.dump_path.with_suffix(config.dump_path.suffix + ".part")
    logger.info("Downloading Croatian Wikipedia dump from %s", config.source.dump_url)

    with urllib.request.urlopen(str(config.source.dump_url)) as response, temporary_path.open("wb") as destination:
        shutil.copyfileobj(response, destination, length=config.runtime.chunk_size_bytes)

    temporary_path.replace(config.dump_path)
    logger.info("Stored dump at %s", config.dump_path)
    return config.dump_path


def _local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def _page_record(page: element_tree.Element) -> tuple[str, str, str, bool]:
    title = ""
    namespace = ""
    text = ""
    has_redirect = False

    for node in page.iter():
        local_name = _local_name(node.tag)
        if local_name == "title" and not title:
            title = node.text or ""
        elif local_name == "ns" and not namespace:
            namespace = node.text or ""
        elif local_name == "redirect":
            has_redirect = True
        elif local_name == "text":
            text = node.text or ""

    return title, namespace, text, has_redirect


def iter_croatian_articles(dump_path: Path, article_limit: int | None = None) -> list[str]:
    article_texts: list[str] = []
    processed_articles = 0

    with bz2.open(dump_path, mode="rt", encoding="utf-8") as compressed_stream:
        context = element_tree.iterparse(compressed_stream, events=("end",))
        for _, element in context:
            if _local_name(element.tag) != "page":
                continue

            title, namespace, text, has_redirect = _page_record(element)
            if namespace == "0" and not has_redirect:
                normalized_text = normalize_croatian_text(text)
                if normalized_text:
                    article_texts.append(normalized_text)
                    processed_articles += 1
                    if article_limit is not None and processed_articles >= article_limit:
                        element.clear()
                        break

            element.clear()

    return article_texts
