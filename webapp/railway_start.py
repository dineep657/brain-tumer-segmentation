"""Backward-compatible entrypoint — delegates to start.py. """

from webapp.start import main

if __name__ == "__main__":
    main()
