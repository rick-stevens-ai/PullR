#!/usr/bin/env bash
# pullr_and_ingest.sh — run PullR and feed the output into SCOUT in one shot.
#
# Usage:
#   pullr_and_ingest.sh <refs-file-or-pdf> [pullr-args...]
#
# Examples:
#   ./pullr_and_ingest.sh /tmp/dois.txt
#   ./pullr_and_ingest.sh ~/some_review.pdf --mode pdf
#   ./pullr_and_ingest.sh refs.txt --model llama --mode exact
#
# The script:
#   1. Picks a fresh output directory under ~/Dropbox/PullR/runs/<ts>/
#   2. Activates PullR's venv
#   3. Runs pullr.py with whatever args you pass (defaults to --mode exact)
#   4. Ingests the output into SCOUT via ingest_to_scout.py
#   5. Tags everything with corpus="pullr-run-<ts>"
#   6. Reports gap-fill stats.

set -euo pipefail

PULLR_DIR="${PULLR_DIR:-$HOME/Dropbox/PullR}"
SCOUT_DIR="${SCOUT_DIR:-$HOME/paper-index}"
SCOUT_DB="${SCOUT_DB:-$SCOUT_DIR/data/papers.sqlite}"
VENV="${VENV:-$PULLR_DIR/.venv}"

if [ $# -lt 1 ]; then
    echo "usage: $0 <refs-file-or-pdf> [extra pullr args...]" >&2
    exit 1
fi

INPUT="$1"; shift
if [ ! -e "$INPUT" ]; then
    echo "error: input '$INPUT' does not exist" >&2
    exit 2
fi

# Detect mode/model: prepend defaults if not supplied. Use $@ directly to
# avoid the empty-element pitfall when no extra args were given.
HAS_MODE=false
HAS_MODEL=false
for a in "$@"; do
    [[ "$a" == "--mode"  || "$a" == "--mode="*  ]] && HAS_MODE=true
    [[ "$a" == "--model" || "$a" == "--model="* ]] && HAS_MODEL=true
done
EXTRA_ARGS=()
if [ "$HAS_MODEL" = "false" ]; then
    EXTRA_ARGS+=("--model" "llama")
fi
if [ "$HAS_MODE" = "false" ]; then
    case "$INPUT" in
        *.pdf|*.PDF) EXTRA_ARGS+=("--mode" "pdf") ;;
        *)           EXTRA_ARGS+=("--mode" "exact") ;;
    esac
fi
EXTRA_ARGS+=("$@")

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$PULLR_DIR/runs/$TS"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"

echo "════════════════════════════════════════════════════════════════"
echo " PullR + SCOUT pipeline"
echo "════════════════════════════════════════════════════════════════"
echo " input        : $INPUT"
echo " output dir   : $OUT_DIR"
echo " pullr args   : ${EXTRA_ARGS[*]}"
echo " log          : $LOG"
echo " scout db     : $SCOUT_DB"
echo "════════════════════════════════════════════════════════════════"

# --- 1. Pre-ingest snapshot
PRE_COUNT=$(sqlite3 "$SCOUT_DB" "SELECT COUNT(*) FROM papers")
echo "[pre]  scout papers: $PRE_COUNT"

# --- 2. Run PullR
echo "[pullr] starting…"
cd "$PULLR_DIR"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
PULLR_SKIP_PREPROCESS="${PULLR_SKIP_PREPROCESS:-1}" \
  python -u pullr.py "$INPUT" \
    --output-dir "$OUT_DIR" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG"

# --- 3. Quick result summary
echo
echo "[pullr] done. Output summary:"
PDF_COUNT=$(find "$OUT_DIR" -maxdepth 1 -type f -iname "*.pdf" | wc -l | tr -d ' ')
TXT_COUNT=$(find "$OUT_DIR" -maxdepth 1 -type f -iname "*.txt" | grep -cv "cleaned_references\|extracted_references" || true)
echo "   PDFs:      $PDF_COUNT"
echo "   abstracts: $TXT_COUNT"

# --- 4. Ingest into SCOUT
echo
echo "[ingest] feeding output into SCOUT…"
python "$SCOUT_DIR/scripts/ingest_pullr_to_scout.py" \
    "$OUT_DIR" \
    --corpus-prefix "pullr-run-$TS"

# --- 5. Post-ingest stats
POST_COUNT=$(sqlite3 "$SCOUT_DB" "SELECT COUNT(*) FROM papers")
DELTA=$((POST_COUNT - PRE_COUNT))
echo
echo "════════════════════════════════════════════════════════════════"
echo " RESULT"
echo "════════════════════════════════════════════════════════════════"
echo "   scout papers before:  $PRE_COUNT"
echo "   scout papers after:   $POST_COUNT"
echo "   net new:              +$DELTA"
echo "   pullr run output:     $OUT_DIR"
echo "════════════════════════════════════════════════════════════════"
