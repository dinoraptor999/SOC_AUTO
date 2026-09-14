#!/bin/bash
set -e
python -m src.train
python -m src.evaluate
