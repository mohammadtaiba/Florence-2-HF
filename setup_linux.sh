#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .

echo
echo "Einrichtung abgeschlossen."
echo "Lege ein Bild unter images/test.jpg ab."
echo "Beispiel:"
echo ".venv/bin/python -m florence2_hf.cli --image images/test.jpg --task caption"
