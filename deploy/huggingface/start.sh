#!/usr/bin/env bash
# Start the backend privately, wait for it, then serve the dashboard on the public port.
set -euo pipefail

cd /app/backend
/opt/backend-venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 &

echo "Waiting for backend..."
for _ in $(seq 1 120); do
  curl -sf http://127.0.0.1:8000/health > /dev/null && break
  sleep 1
done

cd /app/frontend
# XSRF/CORS off: Spaces serve the app inside an iframe, which otherwise breaks file uploads
/opt/frontend-venv/bin/streamlit run app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --client.showSidebarNavigation false \
  --browser.gatherUsageStats false &

# Exit (so the Space restarts) if either process dies
wait -n
exit 1
