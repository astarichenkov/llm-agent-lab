"""Allow ``python -m app.services.rag`` as a shortcut for the CLI."""
from app.services.rag.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
