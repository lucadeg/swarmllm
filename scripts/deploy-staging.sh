#!/usr/bin/env bash
# Manual fallback only: Vercel's git integration deploys every branch push automatically
# (pooled-dev.vercel.app follows feat/engine-opt; main deploys production at pooled.run).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$(npx -y vercel deploy --yes 2>/dev/null)
URL=$(printf '%s' "$OUT" | python3 -c "
import json,re,sys
d=json.loads(re.search(r'\{.*\}', sys.stdin.read(), re.S).group(0))
u=(d.get('deployment') or {}).get('url','')
print(u if str(u).startswith('http') else 'https://'+str(u))")
npx -y vercel alias set "$URL" pooled-dev.vercel.app
echo "staging: https://pooled-dev.vercel.app"
