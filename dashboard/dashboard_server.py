import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from flask import Flask, render_template, jsonify, Response, stream_with_context, request
from api import get_status, compute_stats, get_positions, get_recent_events, get_trades, get_logs, get_all_data

app = Flask(__name__, template_folder='templates', static_folder='static')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    return jsonify(get_status())


@app.route('/api/stats')
def api_stats():
    return jsonify(compute_stats())


@app.route('/api/positions')
def api_positions():
    return jsonify(get_positions())


@app.route('/api/logs')
def api_logs():
    n = request.args.get('n', 100, type=int)
    return jsonify(get_logs(n))


@app.route('/api/trades')
def api_trades():
    return jsonify(get_trades())


@app.route('/api/stream')
def stream():
    def gen():
        while True:
            try:
                data = get_all_data()
                yield f"data: {json.dumps(data)}\n\n"
            except Exception:
                yield f"data: {json.dumps({'error': 'internal error'})}\n\n"
            time.sleep(1)
    return Response(stream_with_context(gen()), mimetype='text/event-stream')
