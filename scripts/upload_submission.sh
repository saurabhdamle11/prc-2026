#!/usr/bin/env bash
# Upload a submission to the team's MinIO bucket using the MinIO client (mc).
#
#   ./scripts/upload_submission.sh submissions/diligent-fireplace_v1.parquet
#
# The organizers require `mc` — the web console is read-only for teams, which is why the
# object browser shows "Access: R" and its Upload File button is disabled. That is by
# design, not a permissions bug.
#
# ONE-TIME SETUP
#
#   1. Install the client:
#        brew install minio/stable/mc
#
#   2. Generate access keys for your OpenSky account: in the MinIO console sidebar go to
#      "Access Keys" -> Create. The secret is shown only once, so copy it then.
#
#   3. Register the alias (this writes the keys to ~/.mc/config.json, not to this repo):
#        mc alias set prc https://s3.opensky-network.org/ ACCESS_KEY SECRET_KEY
#
# Then this script is all you need for each submission.
#
# Objects go at the BUCKET ROOT. `basename` below enforces that: a key prefix such as
# submissions/<file> is not the naming the scorer looks for.
set -euo pipefail

ALIAS="${PRC_ALIAS:-prc}"
BUCKET="${PRC_BUCKET:-prc-2026-diligent-fireplace}"

FILE="${1:-}"
if [[ -z "$FILE" ]]; then echo "usage: $0 <path-to-parquet>" >&2; exit 2; fi
if [[ ! -f "$FILE" ]]; then echo "no such file: $FILE" >&2; exit 2; fi
if ! command -v mc >/dev/null; then
  echo "mc not found — install it with:  brew install minio/stable/mc" >&2; exit 2
fi
if ! mc alias list "$ALIAS" >/dev/null 2>&1; then
  echo "alias '$ALIAS' is not configured. Run:" >&2
  echo "  mc alias set $ALIAS https://s3.opensky-network.org/ ACCESS_KEY SECRET_KEY" >&2
  exit 2
fi

KEY="$(basename "$FILE")"
# The scorer keys off the filename; a wrong name is silently ignored rather than rejected.
# Leaderboard submissions are _v<N>; the single final-phase submission (ranking.html, "Final
# Submission") must be named exactly _final and built from final_submitting.parquet.
if [[ ! "$KEY" =~ ^diligent-fireplace_(v[0-9]+|final)\.parquet$ ]]; then
  echo "refusing to upload '$KEY' — must match diligent-fireplace_v<N>.parquet or diligent-fireplace_final.parquet" >&2; exit 2
fi

echo "uploading $FILE"
echo "     -> $ALIAS/$BUCKET/$KEY"
mc cp "$FILE" "$ALIAS/$BUCKET/$KEY"

echo
echo "bucket contents now:"
mc ls "$ALIAS/$BUCKET/"
echo
echo "A result file should appear here shortly — that is where the score comes back."
echo "Re-run:  mc ls $ALIAS/$BUCKET/"
