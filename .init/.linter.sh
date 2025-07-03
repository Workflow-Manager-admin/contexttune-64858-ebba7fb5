#!/bin/bash
cd /home/kavia/workspace/code-generation/contexttune-64858-ebba7fb5/musemap_backend
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi

