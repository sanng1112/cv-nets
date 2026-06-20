"""cvnets.papers -- arXiv paper search and download toolkit."""
from __future__ import annotations
import dataclasses
import datetime
from pathlib import Path
from typing import List, Optional


@dataclasses.dataclass(frozen=True)
class SearchConfig:
    query: str = "cat:cs.CV"
    max_results: int = 100
    date_from: datetime.date = datetime.date(2015, 1, 1)
    date_to: datetime.date = datetime.date(2026, 4, 30)
    sort_by: str = "submittedDate"
    sort_order: str = "descending"
    download_pdf: bool = True
    output_dir: str = "paper"


@dataclasses.dataclass(frozen=True)
class Paper:
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published: datetime.date
    updated: Optional[datetime.date] = None
    categories: List[str] = dataclasses.field(default_factory=list)
    pdf_url: Optional[str] = None
    abs_url: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.arxiv_id:
            raise ValueError("arxiv_id must not be empty")
        if not self.title:
            raise ValueError("title must not be empty")


from .client import ArxivClient  # noqa
from .storage import PaperStorage  # noqa
from .orchestrator import PaperOrchestrator  # noqa


def search_papers(config: SearchConfig) -> List[Paper]:
    return ArxivClient().search(config)


def download_papers(
    papers: List[Paper], output_dir: str = "paper", pdf: bool = True,
) -> List[Path]:
    return PaperOrchestrator(PaperStorage(output_dir)).download(papers, pdf=pdf)


__all__ = [
    "Paper", "SearchConfig", "search_papers", "download_papers",
    "ArxivClient", "PaperStorage", "PaperOrchestrator",
]
