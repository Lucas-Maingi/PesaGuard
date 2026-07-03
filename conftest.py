import os
import sys

# Ensure the repository root is importable so tests can `import src.*`
# regardless of how pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
