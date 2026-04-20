#!/usr/bin/env python
"""Backward-compatible wrapper for packaged DVAC spectrogram creation."""

from ember.cli import main_spectrograms_dvac as main
from ember.spectrograms import create_dvac_spectrograms as create_spectrograms

__all__ = ["create_spectrograms", "main"]


if __name__ == "__main__":
    main()
