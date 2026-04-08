"""Tests for the document processor module."""

import os
import tempfile
import unittest
from pathlib import Path

from app.core.document_processor import DocumentProcessor, Document


class TestDocumentProcessor(unittest.TestCase):
    """Test suite for DocumentProcessor."""

    def setUp(self) -> None:
        self.processor = DocumentProcessor()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        # Clean up temp files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_temp_file(self, filename: str, content: str) -> str:
        """Create a temporary file with given content."""
        path = os.path.join(self.temp_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_process_txt_file(self) -> None:
        """Test processing a plain text file."""
        content = "This is a test document.\nIt has multiple lines.\nThird line here."
        path = self._create_temp_file("test.txt", content)

        result = self.processor.process(path)

        self.assertIsInstance(result, Document)
        self.assertIn("This is a test document", result.content)
        self.assertEqual(result.metadata["filename"], "test.txt")
        self.assertEqual(result.metadata["format"], ".txt")

    def test_process_markdown_file(self) -> None:
        """Test processing a Markdown file."""
        content = "# Heading\n\nSome paragraph text.\n\n## Subheading\n\nMore text."
        path = self._create_temp_file("test.md", content)

        result = self.processor.process(path)

        self.assertIsInstance(result, Document)
        self.assertIn("Heading", result.content)
        self.assertIn("Some paragraph text", result.content)

    def test_process_empty_file(self) -> None:
        """Test processing an empty text file."""
        path = self._create_temp_file("empty.txt", "")

        result = self.processor.process(path)

        self.assertIsInstance(result, Document)
        self.assertEqual(result.content.strip(), "")

    def test_process_nonexistent_file(self) -> None:
        """Test processing a file that doesn't exist raises an error."""
        with self.assertRaises(Exception):
            self.processor.process("/nonexistent/path/file.txt")

    def test_process_unsupported_format(self) -> None:
        """Test processing an unsupported file format."""
        path = self._create_temp_file("test.xyz", "some content")

        with self.assertRaises(Exception):
            self.processor.process(path)

    def test_document_metadata(self) -> None:
        """Test that document metadata is populated correctly."""
        content = "Test content for metadata check."
        path = self._create_temp_file("metadata_test.txt", content)

        result = self.processor.process(path)

        self.assertEqual(result.metadata["filename"], "metadata_test.txt")
        self.assertEqual(result.metadata["format"], ".txt")
        self.assertIn("file_size", result.metadata)
        self.assertIn("date_processed", result.metadata)
        self.assertGreater(result.metadata["file_size"], 0)

    def test_document_dataclass_fields(self) -> None:
        """Test Document dataclass has expected fields."""
        doc = Document(
            content="test",
            pages=["page1"],
            metadata={"filename": "test.txt", "format": ".txt"},
        )
        self.assertEqual(doc.content, "test")
        self.assertEqual(doc.pages, ["page1"])
        self.assertEqual(doc.metadata["filename"], "test.txt")

    def test_process_csv_file(self) -> None:
        """Test processing a CSV file."""
        content = "name,score,grade\nAlice,95,A\nBob,87,B\nCharlie,72,C"
        path = self._create_temp_file("grades.csv", content)

        result = self.processor.process(path)

        self.assertIsInstance(result, Document)
        self.assertIn("Alice", result.content)
        self.assertIn("Bob", result.content)

    def test_process_html_file(self) -> None:
        """Test processing an HTML file."""
        content = "<html><body><h1>Title</h1><p>Paragraph text.</p></body></html>"
        path = self._create_temp_file("page.html", content)

        result = self.processor.process(path)

        self.assertIsInstance(result, Document)
        self.assertIn("Title", result.content)
        self.assertIn("Paragraph text", result.content)

    def test_unicode_content(self) -> None:
        """Test processing files with Unicode characters."""
        content = "Zurich is a city. Schrodinger's equation: E = hv. Cafe au lait."
        path = self._create_temp_file("unicode.txt", content)

        result = self.processor.process(path)

        self.assertIn("Zurich", result.content)


if __name__ == "__main__":
    unittest.main()
