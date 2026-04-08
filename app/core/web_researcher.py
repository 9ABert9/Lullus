"""
Web research module for Lullus.

Searches the web using DuckDuckGo for academic and educational content
related to course topics. No API key required.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single web search result."""
    title: str
    url: str
    snippet: str
    source_domain: str


class WebResearcher:
    """Searches the web for educational resources using DuckDuckGo."""

    ACADEMIC_DOMAINS = [
        "arxiv.org",
        "scholar.google.com",
        "wikipedia.org",
        "coursera.org",
        "mit.edu",
        "stanford.edu",
        "khanacademy.org",
        "nature.com",
        "sciencedirect.com",
        "springer.com",
        "ieee.org",
        "acm.org",
    ]

    def __init__(self, llm_engine=None, max_results: int = 5) -> None:
        self.llm_engine = llm_engine
        self.max_results = max_results

    def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        """Search the web using DuckDuckGo.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.

        Returns:
            List of SearchResult objects.
        """
        limit = max_results or self.max_results
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.error("duckduckgo-search package not installed")
            return []

        results: List[SearchResult] = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=limit):
                    domain = urlparse(r.get("href", "")).netloc
                    results.append(SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        snippet=r.get("body", ""),
                        source_domain=domain,
                    ))
        except Exception as e:
            logger.error("DuckDuckGo search failed: %s", e)

        return results

    def search_academic(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        """Search for academic/educational content, filtering to known academic domains.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.

        Returns:
            List of SearchResult objects from academic sources.
        """
        limit = max_results or self.max_results
        # Fetch more results so we can filter and still have enough
        all_results = self.search(query, max_results=limit * 3)

        academic_results = [
            r for r in all_results
            if any(domain in r.source_domain for domain in self.ACADEMIC_DOMAINS)
        ]

        # If not enough academic results, also try with site-specific queries
        if len(academic_results) < limit:
            try:
                from duckduckgo_search import DDGS
                with DDGS() as ddgs:
                    site_query = f"{query} site:arxiv.org OR site:wikipedia.org OR site:scholar.google.com"
                    for r in ddgs.text(site_query, max_results=limit):
                        domain = urlparse(r.get("href", "")).netloc
                        result = SearchResult(
                            title=r.get("title", ""),
                            url=r.get("href", ""),
                            snippet=r.get("body", ""),
                            source_domain=domain,
                        )
                        # Avoid duplicates by URL
                        if result.url not in {ar.url for ar in academic_results}:
                            academic_results.append(result)
            except Exception as e:
                logger.warning("Academic site search failed: %s", e)

        return academic_results[:limit]

    def synthesize_results(
        self,
        query: str,
        results: List[SearchResult],
        profile: dict,
    ) -> str:
        """Use the LLM to create a readable summary of search results.

        Args:
            query: The original search query.
            results: List of search results to synthesize.
            profile: Student profile dict.

        Returns:
            A formatted summary string.
        """
        if not results:
            return "No results found for your query."

        if not self.llm_engine:
            return self._format_results_plain(results)

        student_name = profile.get("student", {}).get("name", "Student")
        course_name = profile.get("course", {}).get("name", "the course")
        language = profile.get("course", {}).get("language", "english")

        results_text = ""
        for i, r in enumerate(results, 1):
            results_text += (
                f"{i}. **{r.title}**\n"
                f"   URL: {r.url}\n"
                f"   {r.snippet}\n\n"
            )

        system_prompt = (
            f"You are a research assistant helping {student_name} find resources "
            f"for '{course_name}'. Summarize web search results clearly and helpfully. "
            f"Respond in {language}."
        )

        user_prompt = (
            f"I searched for: \"{query}\"\n\n"
            f"Here are the results:\n\n{results_text}\n"
            f"Please provide a brief, helpful summary of these resources. "
            f"Highlight which ones are most relevant and why. "
            f"Organize by usefulness to a student studying this topic."
        )

        try:
            return self.llm_engine.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.5,
                max_tokens=1024,
            )
        except Exception as e:
            logger.error("LLM synthesis failed: %s", e)
            return self._format_results_plain(results)

    @staticmethod
    def _format_results_plain(results: List[SearchResult]) -> str:
        """Format results as plain text without LLM synthesis."""
        lines = ["Here are the resources I found:\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. **{r.title}**")
            lines.append(f"   {r.snippet}")
            lines.append(f"   Link: {r.url}\n")
        return "\n".join(lines)
