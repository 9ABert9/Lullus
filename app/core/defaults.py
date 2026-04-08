"""
Default configuration for Lullus.

Replaces the old profile-based system with a simple, static config dict.
All modules that previously required a student profile now use DEFAULT_CONFIG.
"""

DEFAULT_CONFIG: dict = {
    "language": "english",
    "citation_style": "APA",
    "code_language": "python",
    "verbosity": "standard",
    "tone": "friendly and professional",
    "knowledge_level": "intermediate",
    "output_format": "markdown",
}
