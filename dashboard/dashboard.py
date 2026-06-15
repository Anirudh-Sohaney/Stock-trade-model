#!/usr/bin/env python
"""Dashboard entry point. Starts the web dashboard on 0.0.0.0:11434."""

from dashboard_server import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=11434, debug=False, use_reloader=False)
