"""arXiv API client wrapping the ``arxiv`` Python library."""
from __future__ import annotations
import datetime
import logging
import re
from pathlib import Path
from typing import List
import arxiv
import requests
from . import Paper, SearchConfig

logger = logging.getLogger(__name__)


class ArxivClient:

    def search(self, config: SearchConfig) -> List[Paper]:
        sc = self._resolve_sort(config.sort_by)
        so = self._resolve_order(config.sort_order)
        search = arxiv.Search(
            query=config.query, max_results=config.max_results,
            sort_by=sc, sort_order=so,
        )
        papers: List[Paper] = []
        for result in arxiv.Client().results(search):
            pub_date = result.published.date() if result.published else None
            if pub_date is None:
                continue
            if pub_date < config.date_from or pub_date > config.date_to:
                continue
            papers.append(self._result_to_paper(result))
        return papers

    def download_pdf(self, paper: Paper, dest: Path) -> Path:
        if not paper.pdf_url:
            raise ValueError(f"No PDF URL for paper {paper.arxiv_id}")
        response = requests.get(paper.pdf_url, timeout=30)
        response.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(response.content)
        return dest

    def _result_to_paper(self, r: arxiv.Result) -> Paper:
        arxiv_id = self._extract_id(r.entry_id)
        authors = [a.name for a in r.authors]
        cats = list(r.categories) if r.categories else []
        pdf_url = str(r.pdf_url) if r.pdf_url else None
        abs_url = str(r.entry_id) if r.entry_id else None
        if not pdf_url and r.links:
            for link in r.links:
                if link.title == "pdf":
                    pdf_url = str(link.href)
                    break
        return Paper(
            arxiv_id=arxiv_id, title=str(r.title or "").strip(),
            authors=authors, abstract=str(r.summary or "").strip(),
            published=r.published.date() if r.published else datetime.date.min,
            updated=r.updated.date() if r.updated else None,
            categories=cats, pdf_url=pdf_url, abs_url=abs_url,
        )

    @staticmethod
    def _extract_id(entry_id: str) -> str:
        m = re.search(r"/abs/([^v/]+(?:/\d+)?)(?:v\d+)?$", entry_id)
        if m:
            return m.group(1)
        m = re.search(r"/abs/([a-z\-]+(?:\.[A-Z]{2})?/\d+\.\d+)", entry_id)
        if m:
            return m.group(1)
        return entry_id.rstrip("/").split("/")[-1].split("v")[0]

    @staticmethod
    def _resolve_sort(sort_by: str):
        s = {
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        }
        return s.get(sort_by, arxiv.SortCriterion.SubmittedDate)

    @staticmethod
    def _resolve_order(sort_order: str):
        o = {
            "descending": arxiv.SortOrder.Descending,
            "ascending": arxiv.SortOrder.Ascending,
        }
        return o.get(sort_order, arxiv.SortOrder.Descending)
