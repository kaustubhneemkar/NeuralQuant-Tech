"""
Shared test configuration.
Mocks google.genai before any test module imports it.
"""
import sys
import os
from unittest.mock import MagicMock

# Mock google.genai SDK so tests don't require the full package
genai_mock = MagicMock()
genai_mock.Client = MagicMock
sys.modules.setdefault("google.genai", genai_mock)

# Ensure API key is set
os.environ.setdefault("GOOGLE_API_KEY", "dummy-key-for-testing")
