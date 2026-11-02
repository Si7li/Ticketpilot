#!/bin/sh
# Seed the business DB if missing and index the KB if the Qdrant collection is missing/stale,
# then run the given command. Both steps are idempotent.
set -e
if [ "${SKIP_BOOTSTRAP:-0}" != "1" ]; then
  if [ ! -f "${BUSINESS_DB_PATH}" ]; then
    echo "entrypoint: seeding ${BUSINESS_DB_PATH}"
    python /app/data/seed/seed_db.py --no-scenarios --db "${BUSINESS_DB_PATH}"
  fi
  if [ -n "${QDRANT_URL}" ]; then
    echo "entrypoint: ensuring KB is indexed in ${QDRANT_URL}"
    python /app/scripts/index_kb.py
  fi
fi
exec "$@"
