#!/usr/bin/env python
"""Backward-compatible wrapper for packaged VAC spectrogram creation."""

from ember.cli import main_spectrograms_vac as main
from ember.spectrograms import create_vac_spectrograms as create_spectrograms

__all__ = ["create_spectrograms", "main"]


if __name__ == "__main__":
    main()
