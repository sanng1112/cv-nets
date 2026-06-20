"""Tests for ArxivClient."""
from __future__ import annotations
import datetime
from unittest.mock import MagicMock, patch
import pytest
from cvnets.papers import SearchConfig
from cvnets.papers.client import ArxivClient


class TestArxivClient:

    def test_search_returns_list_of_papers(self):
        mock_result = MagicMock()
        mock_result.entry_id = "http://arxiv.org/abs/1512.03385v1"
        mock_result.title = "Deep Residual Learning for Image Recognition"
        author_mock = MagicMock()
        author_mock.name = "Kaiming He"
        mock_result.authors = [author_mock]
        mock_result.summary = "Deep residual networks..."
        mock_result.published = datetime.datetime(2015, 12, 10)
        mock_result.updated = datetime.datetime(2016, 1, 1)
        mock_result.categories = ["cs.CV"]
        mock_result.pdf_url = "http://arxiv.org/pdf/1512.03385v1"
        mock_result.links = [
            MagicMock(href="http://arxiv.org/abs/1512.03385v1", title="abs"),
            MagicMock(href="http://arxiv.org/pdf/1512.03385v1", title="pdf"),
        ]
        with patch("arxiv.Search") as mock_search_cls:
            with patch("arxiv.Client") as mock_client_cls:
                mock_search = MagicMock()
                mock_search_cls.return_value = mock_search
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client
                mock_client.results.return_value = [mock_result]
                client = ArxivClient()
                config = SearchConfig(
                    query="cat:cs.CV", max_results=1,
                    date_from=datetime.date(2015, 1, 1),
                    date_to=datetime.date(2016, 12, 31),
                )
                papers = client.search(config)
        assert len(papers) == 1
        p = papers[0]
        assert p.arxiv_id == "1512.03385"
        assert "Deep Residual Learning" in p.title
        assert p.authors == ["Kaiming He"]
        assert p.published == datetime.date(2015, 12, 10)
        assert p.categories == ["cs.CV"]

    def test_search_empty_response(self):
        with patch("arxiv.Search") as mock_search_cls:
            with patch("arxiv.Client") as mock_client_cls:
                mock_search = MagicMock()
                mock_search_cls.return_value = mock_search
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client
                mock_client.results.return_value = []
                client = ArxivClient()
                papers = client.search(SearchConfig(max_results=10))
        assert papers == []

    def test_search_applies_date_filter(self):
        old = MagicMock()
        old.entry_id = "http://arxiv.org/abs/1401.0001v1"
        old.title = "Old"
        old_author = MagicMock()
        old_author.name = "A"
        old.authors = [old_author]
        old.summary = "Abstract"
        old.published = datetime.datetime(2014, 1, 1)
        old.updated = None
        old.categories = ["cs.CV"]
        old.pdf_url = None
        old.links = []
        new = MagicMock()
        new.entry_id = "http://arxiv.org/abs/1512.03385v1"
        new.title = "New"
        new_author = MagicMock()
        new_author.name = "B"
        new.authors = [new_author]
        new.summary = "Abstract"
        new.published = datetime.datetime(2015, 12, 10)
        new.updated = None
        new.categories = ["cs.CV"]
        new.pdf_url = None
        new.links = []
        with patch("arxiv.Search") as mock_search_cls:
            with patch("arxiv.Client") as mock_client_cls:
                mock_search = MagicMock()
                mock_search_cls.return_value = mock_search
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client
                mock_client.results.return_value = [old, new]
                client = ArxivClient()
                papers = client.search(SearchConfig(
                    max_results=10,
                    date_from=datetime.date(2015, 1, 1),
                    date_to=datetime.date(2016, 12, 31),
                ))
        assert len(papers) == 1
        assert papers[0].arxiv_id == "1512.03385"
