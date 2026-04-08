"""
File system watcher for Lullus.

Monitors the knowledge_base/ folder for new, modified, or deleted files
and automatically triggers indexing via the embedding manager.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".md", ".epub", ".html", ".csv"}


@dataclass
class WatcherEvent:
    """A single file watcher event."""
    timestamp: str
    event_type: str  # created, modified, deleted
    file_path: str
    status: str  # success, error, pending
    message: str = ""


class KnowledgeBaseHandler(FileSystemEventHandler):
    """Handles file system events in the knowledge base folder."""

    def __init__(self, embedding_manager, document_processor) -> None:
        super().__init__()
        self.embedding_manager = embedding_manager
        self.document_processor = document_processor
        self.event_log: Deque[WatcherEvent] = deque(maxlen=50)
        self.files_processed: int = 0
        self.last_event_time: Optional[str] = None
        self._lock = threading.Lock()

    def _is_supported(self, path: str) -> bool:
        return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS

    def _log_event(self, event_type: str, file_path: str, status: str, message: str = "") -> None:
        now = datetime.now().isoformat(timespec="seconds")
        self.last_event_time = now
        self.event_log.appendleft(WatcherEvent(
            timestamp=now,
            event_type=event_type,
            file_path=str(file_path),
            status=status,
            message=message,
        ))

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory or not self._is_supported(event.src_path):
            return
        logger.info("New file detected: %s", event.src_path)
        self._process_file(event.src_path, "created")

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory or not self._is_supported(event.src_path):
            return
        logger.info("File modified: %s", event.src_path)
        self._process_file(event.src_path, "modified")

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory or not self._is_supported(event.src_path):
            return
        logger.info("File deleted: %s", event.src_path)
        self._remove_file(event.src_path)

    def _process_file(self, file_path: str, event_type: str) -> None:
        # Small delay to ensure file is fully written
        time.sleep(0.5)
        with self._lock:
            try:
                self.embedding_manager.add_document(file_path)
                self.files_processed += 1
                self._log_event(event_type, file_path, "success",
                                f"Indexed {Path(file_path).name}")
                logger.info("Successfully indexed: %s", file_path)
            except Exception as e:
                self._log_event(event_type, file_path, "error", str(e))
                logger.error("Failed to index %s: %s", file_path, e)

    def _remove_file(self, file_path: str) -> None:
        with self._lock:
            try:
                filename = Path(file_path).name
                # Find and remove document by filename
                all_docs = self.embedding_manager.get_all_documents()
                for doc in all_docs:
                    if doc.filename == filename:
                        self.embedding_manager.remove_document(doc.doc_id)
                        self._log_event("deleted", file_path, "success",
                                        f"Removed {filename} from index")
                        logger.info("Removed from index: %s", file_path)
                        return
                self._log_event("deleted", file_path, "success",
                                f"{filename} was not in the index")
            except Exception as e:
                self._log_event("deleted", file_path, "error", str(e))
                logger.error("Failed to remove %s from index: %s", file_path, e)


class FileWatcher:
    """Watches the knowledge_base/ folder and auto-indexes files."""

    def __init__(self, watch_path: str, embedding_manager, document_processor) -> None:
        self.watch_path = Path(watch_path).resolve()
        self.watch_path.mkdir(parents=True, exist_ok=True)
        self.handler = KnowledgeBaseHandler(embedding_manager, document_processor)
        self.observer: Optional[Observer] = None
        self._running = False

    def start(self) -> None:
        """Start watching the knowledge base folder."""
        if self._running:
            logger.info("File watcher is already running")
            return

        self.observer = Observer()
        self.observer.schedule(self.handler, str(self.watch_path), recursive=False)
        self.observer.daemon = True
        self.observer.start()
        self._running = True
        logger.info("File watcher started on: %s", self.watch_path)

    def stop(self) -> None:
        """Stop watching the knowledge base folder."""
        if self.observer and self._running:
            self.observer.stop()
            self.observer.join(timeout=5)
            self._running = False
            logger.info("File watcher stopped")

    def get_status(self) -> Dict:
        """Get current watcher status.

        Returns:
            Dict with is_running, files_processed, last_event, watch_path.
        """
        return {
            "is_running": self._running,
            "files_processed": self.handler.files_processed,
            "last_event": self.handler.last_event_time,
            "watch_path": str(self.watch_path),
        }

    def get_log(self) -> List[Dict]:
        """Get the event log (last 50 events).

        Returns:
            List of event dicts ordered most recent first.
        """
        return [
            {
                "timestamp": e.timestamp,
                "event_type": e.event_type,
                "file_path": e.file_path,
                "status": e.status,
                "message": e.message,
            }
            for e in self.handler.event_log
        ]
