#!/usr/bin/env bash
# Setup script: installs dev dependencies and git pre-commit hook.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOOK="$REPO_ROOT/.git/hooks/pre-commit"

echo "Installing dev dependencies..."
pip install -e "$REPO_ROOT[dev]"

echo "Installing pre-commit hook..."
cat > "$HOOK" << 'EOF'
#!/usr/bin/env bash
# Pre-commit hook: reject commits with unformatted Python files.
set -euo pipefail

STAGED=$(git diff --cached --name-only --diff-filter=ACM -- '*.py')
if [ -z "$STAGED" ]; then
    exit 0
fi

if ! .venv/bin/black --check --quiet $STAGED; then
    echo "ERROR: Files not formatted with black. Run 'black .' and re-stage."
    exit 1
fi
EOF
chmod +x "$HOOK"

echo "Done. Pre-commit hook installed at $HOOK"
