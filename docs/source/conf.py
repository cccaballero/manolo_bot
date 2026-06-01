import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "manolo-bot"
copyright = "2024, Carlos Cesar Caballero Díaz"
author = "Carlos Cesar Caballero Díaz"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autodoc_mock_imports = [
    "redis",
    "pypdf",
    "docx",
    "aiohttp",
    "google",
    "langchain",
    "langchain_classic",
    "langchain_community",
    "langchain_core",
    "langchain_google_genai",
    "langchain_ollama",
    "langchain_openai",
    "pydantic",
    "pydantic_settings",
    "youtube_transcript_api",
    "ddgs",
    "llmagent",
    "langchain_mcp_adapters",
    "mcp",
    "langgraph",
]
