"""
Document Loaders - Fetch Wikipedia articles for programming languages

Loads two types of content:
1. Language pages: Python, Java, C++, etc. (from TIOBE top 50)
2. Conceptual pages: Compiler, Type system, etc. (bridge topics)
"""

import asyncio
import logging
import time

import aiohttp
from aiohttp import ClientTimeout
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.documents import Document

from ragchain.utils import log_with_prefix

logger = logging.getLogger(__name__)


# ============================================================================
# Conceptual Bridge Pages (Add cross-language context)
# ============================================================================

# These topics help answer questions that span multiple languages
# Example: "What is a compiler?" needs context beyond individual languages
CONCEPTUAL_TOPICS = [
    "Programming language",  # What is a programming language?
    "Programming language implementation",  # Compiled vs Interpreted
    "Programming paradigm",  # Imperative, Functional, OOP
    "Type system",  # Static vs Dynamic typing
    "Memory management",  # Garbage Collection vs Manual
    "History of programming languages",
    "Compiler",
    "Interpreter (computing)",
    "Standard library",
    "Syntax (programming languages)",
]


# ============================================================================
# TIOBE Language Fetcher
# ============================================================================


async def load_tiobe_languages(n: int = 50) -> list[str]:
    """
    Fetch top programming languages from TIOBE index

    Returns: List of language names (e.g., ['Python', 'C', 'Java', ...])
    """
    url = "https://www.tiobe.com/tiobe-index/"

    # Fetch TIOBE webpage
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=ClientTimeout(total=15)) as r:
                r.raise_for_status()
                html = await r.text()
    except Exception as e:
        log_with_prefix(logger, logging.WARNING, "load_tiobe_languages", f"Failed to fetch TIOBE: {e}")
        return []

    # Parse HTML to extract language names
    soup = BeautifulSoup(html, "html.parser")
    languages = []

    # Extract from top 20 table
    top20_table = soup.find("table", id="top20")
    if top20_table:
        for row in top20_table.find_all("tr")[1:]:  # Skip header row
            cols = row.find_all("td")
            if len(cols) > 4 and (name := cols[4].get_text(strip=True)):
                languages.append(name)

    # Extract from languages 21-50 table
    other_table = soup.find("table", id="otherPL")
    if other_table:
        for row in other_table.find_all("tr")[1:]:  # Skip header row
            cols = row.find_all("td")
            if len(cols) > 1 and (name := cols[1].get_text(strip=True)):
                languages.append(name)

    # Return top n languages
    return languages[:n]


# ============================================================================
# Wikipedia Page Loaders
# ============================================================================


def _load_single_page(lang: str, retries: int = 2) -> Document | None:
    """
    Load Wikipedia article for a programming language (with retries)

    Example: "Python" → Loads "Python (programming language)" article
    """
    for attempt in range(retries + 1):
        try:
            # Add "programming language" to query for better results
            loader = WikipediaLoader(query=f"{lang} programming language", load_max_docs=1)
            pages = loader.load()

            if pages:
                # Add language name to metadata
                pages[0].metadata["language"] = lang
                return pages[0]

        except Exception as e:
            if attempt < retries:
                # Wait before retrying (exponential backoff: 0.5s, 1s, 2s)
                wait_time = 0.5 * (2**attempt)
                time.sleep(wait_time)
            else:
                log_with_prefix(logger, logging.WARNING, "load_wikipedia_page", f"Failed after {retries + 1} attempts: {lang} - {e}")

    return None


def _load_topic_page(topic: str, retries: int = 2) -> Document | None:
    """
    Load Wikipedia article for a conceptual topic (with retries)

    Example: "Compiler" → Loads "Compiler" article
    """
    for attempt in range(retries + 1):
        try:
            # Use exact topic name (no modification)
            loader = WikipediaLoader(query=topic, load_max_docs=1)
            pages = loader.load()

            if pages:
                # Tag as conceptual topic
                pages[0].metadata["category"] = "concept"
                pages[0].metadata["topic"] = topic
                return pages[0]

        except Exception as e:
            if attempt < retries:
                # Exponential backoff
                wait_time = 0.5 * (2**attempt)
                time.sleep(wait_time)
            else:
                log_with_prefix(logger, logging.WARNING, "load_topic_page", f"Failed to load {topic}: {e}")

    return None


# ============================================================================
# Batch Loading Functions
# ============================================================================


async def load_wikipedia_pages(language_names: list[str]) -> list[Document]:
    """
    Load Wikipedia articles for multiple programming languages

    Loads sequentially (one at a time) to avoid Wikipedia rate limits.
    Failed languages are silently skipped.

    Returns: List of Document objects
    """
    docs = []
    loop = asyncio.get_event_loop()

    # Load pages one by one (sequential to avoid rate limiting)
    for lang in language_names:
        try:
            # Run blocking Wikipedia call in executor
            result = await loop.run_in_executor(None, _load_single_page, lang)
            if result:
                docs.append(result)
        except Exception as e:
            log_with_prefix(logger, logging.ERROR, "load_wikipedia_pages", f"Error loading {lang}: {e}")

    return docs


async def load_conceptual_pages() -> list[Document]:
    """
    Load conceptual/theory Wikipedia pages

    Loads the 10 pre-defined CONCEPTUAL_TOPICS that provide
    cross-language context (e.g., "Compiler", "Type system").

    Returns: List of Document objects
    """
    docs = []
    loop = asyncio.get_event_loop()

    logger.info(f"Loading {len(CONCEPTUAL_TOPICS)} conceptual pages...")

    for topic in CONCEPTUAL_TOPICS:
        try:
            # Run blocking Wikipedia call in executor
            result = await loop.run_in_executor(None, _load_topic_page, topic)
            if result:
                docs.append(result)
                logger.info(f"✓ Loaded: {topic}")
        except Exception as e:
            log_with_prefix(logger, logging.ERROR, "load_conceptual_pages", f"Error loading {topic}: {e}")

    return docs
