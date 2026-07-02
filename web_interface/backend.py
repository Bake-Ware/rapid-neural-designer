"""
RND Platform Backend
Serves the visual editor (rooms, code execution) and the research platform API
(programs, projects, threads, experiments, findings, architectures, papers, disclosures).
"""

import os
import re
import secrets
import sys
import subprocess
import tempfile
import time
import json
import random
from pathlib import Path
from flask import Flask, request, jsonify, redirect, send_from_directory, render_template_string
from flask_cors import CORS
from flask_socketio import SocketIO, join_room, leave_room

# Add parent dir to path so we can import the rnd package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rnd.canonical import canonicalize, content_hash
from rnd.index import DerivedIndex
from rnd.models import (
    Architecture, Artifact, ArtifactCategory, Citation,
    Disclosure, DisclosureType, EvidenceLink, EvidenceSign, EvidenceStrength,
    Experiment, ExperimentInputs, ExperimentStatus,
    Finding, FindingResolution, ObservedResults,
    Paper, PaperStatus, PaperSection, SectionType,
    Program, Project, Statement, StatementResolution,
    Team, Thread, ThreadState, Visibility, now_iso, generate_id,
)
from rnd.repo import RNDRepo
from rnd.auth import AuthDB
from rnd.mcp_endpoint import mcp_bp, init_mcp
from functools import wraps

STATIC_DIR = Path(__file__).resolve().parent

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

auth_db: AuthDB = None  # type: ignore


def get_current_user():
    """Extract user from Authorization header. Returns user dict or None."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        return auth_db.validate_token(token) or auth_db.validate_api_token(token)
    return None


def require_auth(f):
    """Decorator: require valid auth token for a route."""
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        request.user = user
        return f(*args, **kwargs)
    return decorated


@app.before_request
def protect_rnd_routes():
    """All /api/rnd/* routes require authentication."""
    if request.path.startswith('/api/rnd/'):
        user = get_current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        request.user = user


# ---------------------------------------------------------------------------
# Static file serving (replaces the old python -m http.server 8089)
# ---------------------------------------------------------------------------

@app.route("/")
def serve_index():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/3d.html")
def serve_3d():
    return send_from_directory(STATIC_DIR, "3d.html")

# ---------------------------------------------------------------------------
# Auth API
# ---------------------------------------------------------------------------

@app.route('/api/auth/register', methods=['POST'])
def auth_register():
    data = request.get_json()
    if not data or not data.get('username') or not data.get('password'):
        return api_error("'username' and 'password' required")
    if len(data['username']) < 3:
        return api_error("Username must be at least 3 characters")
    if len(data['password']) < 4:
        return api_error("Password must be at least 4 characters")
    user = auth_db.register(
        username=data['username'],
        password=data['password'],
        display_name=data.get('display_name', ''),
    )
    if not user:
        return api_error("Username already taken", 409)
    # Auto-login after registration
    result = auth_db.login(data['username'], data['password'])
    return jsonify(result), 201


@app.route('/api/auth/login', methods=['POST'])
def auth_login():
    data = request.get_json()
    if not data or not data.get('username') or not data.get('password'):
        return api_error("'username' and 'password' required")
    result = auth_db.login(data['username'], data['password'])
    if not result:
        return api_error("Invalid username or password", 401)
    return jsonify(result)


@app.route('/api/auth/me', methods=['GET'])
def auth_me():
    user = get_current_user()
    if not user:
        return jsonify({"user": None})
    user["is_admin"] = auth_db.is_admin(user["id"])
    return jsonify({"user": user})


@app.route('/api/auth/logout', methods=['POST'])
def auth_logout():
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        auth_db.logout(auth_header[7:])
    return jsonify({"ok": True})


@app.route('/api/auth/token', methods=['GET'])
@require_auth
def auth_api_token():
    """Return the caller's derived API token (for programmatic / agent / MCP access).

    Derived from the user's credentials (HMAC of user_id:password_hash), so it
    needs no storage and rotates automatically when the password changes.
    """
    token = auth_db.get_api_token(request.user["id"])
    return jsonify({
        "api_token": token,
        "usage": "Authorization: Bearer <api_token>",
        "note": "Stable until you change your password. Treat it like a password.",
    })


@app.route('/api-token')
def serve_api_token_page():
    return send_from_directory(STATIC_DIR, "api_token.html")


# ---------------------------------------------------------------------------
# OAuth 2.0 (for claude.ai MCP connectors)
# ---------------------------------------------------------------------------

@app.route('/authorize', methods=['GET', 'POST'])
def oauth_authorize():
    """OAuth 2.0 Authorization endpoint with PKCE."""
    client_id = request.args.get('client_id', '')
    redirect_uri = request.args.get('redirect_uri', '')
    code_challenge = request.args.get('code_challenge', '')
    code_challenge_method = request.args.get('code_challenge_method', 'S256')
    state = request.args.get('state', '')
    response_type = request.args.get('response_type', '')

    if response_type != 'code':
        return 'Unsupported response_type', 400

    # Verify client_id exists
    client_user = auth_db.get_mcp_client_user(client_id)
    if not client_user:
        return 'Unknown client_id', 400

    if request.method == 'GET':
        # Show login/consent form
        return f'''<!DOCTYPE html>
<html><head><title>RND — Authorize</title>
<style>
  body {{ font-family: system-ui; background: #0a0e1a; color: #e0e0e0;
         display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }}
  .card {{ background: #141824; padding: 2rem; border-radius: 12px; width: 360px;
           box-shadow: 0 4px 24px rgba(0,0,0,0.4); }}
  h2 {{ margin-top: 0; color: #7eb8ff; }}
  p {{ color: #999; font-size: 0.9em; }}
  label {{ display: block; margin-top: 1rem; font-size: 0.85em; color: #bbb; }}
  input {{ width: 100%; padding: 0.6rem; margin-top: 0.3rem; border: 1px solid #333;
           border-radius: 6px; background: #1a1f30; color: #e0e0e0; box-sizing: border-box; }}
  button {{ width: 100%; padding: 0.7rem; margin-top: 1.5rem; border: none; border-radius: 6px;
            background: #3b82f6; color: white; font-size: 1rem; cursor: pointer; }}
  button:hover {{ background: #2563eb; }}
  .error {{ color: #f87171; font-size: 0.85em; margin-top: 0.5rem; }}
</style></head><body>
<div class="card">
  <h2>RND Platform</h2>
  <p>Claude wants to access your research data.</p>
  <form method="POST">
    <input type="hidden" name="client_id" value="{client_id}">
    <input type="hidden" name="redirect_uri" value="{redirect_uri}">
    <input type="hidden" name="code_challenge" value="{code_challenge}">
    <input type="hidden" name="code_challenge_method" value="{code_challenge_method}">
    <input type="hidden" name="state" value="{state}">
    <label>Username<input type="text" name="username" required></label>
    <label>Password<input type="password" name="password" required></label>
    <button type="submit">Authorize</button>
  </form>
</div></body></html>'''

    # POST — validate credentials and issue code
    username = request.form.get('username', '')
    password = request.form.get('password', '')
    client_id = request.form.get('client_id', client_id)
    redirect_uri = request.form.get('redirect_uri', redirect_uri)
    code_challenge = request.form.get('code_challenge', code_challenge)
    code_challenge_method = request.form.get('code_challenge_method', code_challenge_method)
    state = request.form.get('state', state)

    # Authenticate user
    login_result = auth_db.login(username, password)
    if not login_result:
        return f'''<!DOCTYPE html>
<html><head><title>RND — Authorize</title>
<style>
  body {{ font-family: system-ui; background: #0a0e1a; color: #e0e0e0;
         display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }}
  .card {{ background: #141824; padding: 2rem; border-radius: 12px; width: 360px;
           box-shadow: 0 4px 24px rgba(0,0,0,0.4); }}
  h2 {{ margin-top: 0; color: #7eb8ff; }}
  p {{ color: #999; font-size: 0.9em; }}
  label {{ display: block; margin-top: 1rem; font-size: 0.85em; color: #bbb; }}
  input {{ width: 100%; padding: 0.6rem; margin-top: 0.3rem; border: 1px solid #333;
           border-radius: 6px; background: #1a1f30; color: #e0e0e0; box-sizing: border-box; }}
  button {{ width: 100%; padding: 0.7rem; margin-top: 1.5rem; border: none; border-radius: 6px;
            background: #3b82f6; color: white; font-size: 1rem; cursor: pointer; }}
  button:hover {{ background: #2563eb; }}
  .error {{ color: #f87171; font-size: 0.85em; margin-top: 0.5rem; }}
</style></head><body>
<div class="card">
  <h2>RND Platform</h2>
  <p class="error">Invalid username or password.</p>
  <form method="POST">
    <input type="hidden" name="client_id" value="{client_id}">
    <input type="hidden" name="redirect_uri" value="{redirect_uri}">
    <input type="hidden" name="code_challenge" value="{code_challenge}">
    <input type="hidden" name="code_challenge_method" value="{code_challenge_method}">
    <input type="hidden" name="state" value="{state}">
    <label>Username<input type="text" name="username" value="{username}" required></label>
    <label>Password<input type="password" name="password" required></label>
    <button type="submit">Authorize</button>
  </form>
</div></body></html>''', 401

    # Delete the session we just created (we only needed to verify credentials)
    auth_db.logout(login_result['token'])

    # Issue authorization code
    code = auth_db.create_oauth_code(
        client_id=client_id,
        user_id=login_result['user']['id'],
        redirect_uri=redirect_uri,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
    )

    # Redirect back to claude.ai with code
    from urllib.parse import urlencode, urlparse, urlunparse, parse_qs
    params = {'code': code}
    if state:
        params['state'] = state
    separator = '&' if '?' in redirect_uri else '?'
    return redirect(redirect_uri + separator + urlencode(params))


@app.route('/token', methods=['POST'])
def oauth_token():
    """OAuth 2.0 Token endpoint — exchange authorization code for access token."""
    # Accept both form-encoded and JSON
    if request.content_type and 'json' in request.content_type:
        data = request.get_json() or {}
    else:
        data = request.form.to_dict()

    grant_type = data.get('grant_type', '')
    code = data.get('code', '')
    redirect_uri = data.get('redirect_uri', '')
    code_verifier = data.get('code_verifier', '')

    if grant_type != 'authorization_code':
        return jsonify({"error": "unsupported_grant_type"}), 400

    if not code or not code_verifier:
        return jsonify({"error": "invalid_request", "error_description": "code and code_verifier required"}), 400

    result = auth_db.exchange_oauth_code(code, code_verifier, redirect_uri)
    if not result:
        return jsonify({"error": "invalid_grant"}), 400

    return jsonify(result)


@app.route('/.well-known/oauth-authorization-server', methods=['GET'])
def oauth_metadata():
    """OAuth 2.0 Authorization Server Metadata (RFC 8414)."""
    base = request.host_url.rstrip('/')
    return jsonify({
        "issuer": base,
        "authorization_endpoint": f"{base}/authorize",
        "token_endpoint": f"{base}/token",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
    })


# ---------------------------------------------------------------------------
# RND Platform init
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
rnd_repo = RNDRepo(REPO_ROOT)
rnd_index: DerivedIndex | None = None

def init_platform():
    """Initialize the RND repo, auth DB, and derived index on startup."""
    global rnd_index, auth_db
    if not rnd_repo.is_initialized():
        rnd_repo.init(team_name="Bake Research", user_id="user-bake")
        print("[RND] Initialized new repository")
    # Auth database
    auth_db = AuthDB(rnd_repo.rnd_dir / "auth.sqlite")
    auth_db.open()
    print("[RND] Auth ready")
    rnd_index = DerivedIndex(rnd_repo.index_path)
    rnd_index.open()
    rnd_index.rebuild(rnd_repo.root)
    # MCP endpoint
    init_mcp(rnd_repo, auth_db, web_interface_root=STATIC_DIR)
    app.register_blueprint(mcp_bp, url_prefix="/mcp")
    print(f"[RND] MCP endpoint ready at /mcp")
    print(f"[RND] Platform ready — repo at {rnd_repo.root}")

def ensure_index():
    if rnd_index and rnd_index.conn is None:
        rnd_index.open()

def api_error(msg: str, status: int = 400):
    return jsonify({"error": msg}), status

# ---- Room System ----

ADJECTIVES = [
    "farty", "wobbly", "sneaky", "grumpy", "fluffy", "dizzy", "chunky", "sassy",
    "bouncy", "goofy", "cranky", "sparkly", "sleepy", "jumpy", "fuzzy", "spicy",
    "crispy", "zappy", "bubbly", "wiggly", "squishy", "cosmic", "turbo", "mega",
    "hyper", "lazy", "peppy", "zippy", "nutty", "wonky", "chonky", "funky",
    "snazzy", "wacky", "loopy", "dorky", "quirky", "burpy", "giggly", "toasty",
]

NOUNS = [
    "banana", "pickle", "waffle", "noodle", "muffin", "taco", "penguin", "llama",
    "potato", "narwhal", "donut", "burrito", "cactus", "walrus", "platypus",
    "pretzel", "pancake", "avocado", "dumpling", "nugget", "biscuit", "hamster",
    "squid", "gopher", "badger", "otter", "wombat", "gecko", "moose", "yak",
    "tornado", "noodle", "cobbler", "turnip", "kumquat", "blobfish", "quokka",
    "armadillo", "flamingo", "capybara",
]

# In-memory room store: { room_name: { "model": <graph json or None>, "created": <timestamp> } }
rooms = {}


def generate_room_name():
    """Generate a fun two-word room name like 'farty-banana'"""
    for _ in range(100):
        name = f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}"
        if name not in rooms:
            return name
    # Fallback: add random digits
    return f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}-{random.randint(10,99)}"


# ---- Room REST Endpoints ----

@app.route('/api/room', methods=['POST'])
def create_room():
    """Create a new collaboration room. Optionally pass {"name": "my-room"} to choose the name."""
    data = request.json or {}
    name = data.get("name", "").strip().lower()
    if name:
        # Sanitize: only allow alphanumeric, hyphens, underscores
        name = "".join(c for c in name if c.isalnum() or c in "-_")
        if name in rooms:
            # Room already exists, just return it
            return jsonify({"room": name, "existing": True})
    else:
        name = generate_room_name()
    rooms[name] = {
        "model": None,
        "created": time.time(),
    }
    return jsonify({"room": name, "existing": False})


@app.route('/api/room/<room_name>', methods=['GET'])
def get_room(room_name):
    """Get room info"""
    if room_name not in rooms:
        return jsonify({"error": "Room not found"}), 404
    room = rooms[room_name]
    return jsonify({
        "room": room_name,
        "has_model": room["model"] is not None,
        "created": room["created"],
    })


@app.route('/api/room/<room_name>/model', methods=['GET'])
def get_model(room_name):
    """Get the current model graph JSON for a room"""
    if room_name not in rooms:
        return jsonify({"error": "Room not found"}), 404
    return jsonify({
        "room": room_name,
        "model": rooms[room_name]["model"],
    })


@app.route('/api/room/<room_name>/model', methods=['PUT'])
def update_model(room_name):
    """Update the model graph JSON for a room, broadcast to WS clients"""
    if room_name not in rooms:
        return jsonify({"error": "Room not found"}), 404

    data = request.json
    model = data.get("model")
    if model is None:
        return jsonify({"error": "Missing 'model' in request body"}), 400

    rooms[room_name]["model"] = model

    # Broadcast to all WS clients in this room
    socketio.emit("model_updated", {"model": model}, room=room_name)

    return jsonify({"room": room_name, "status": "updated"})


@app.route('/api/rooms', methods=['GET'])
def list_rooms():
    """List all active rooms"""
    return jsonify({
        "rooms": [
            {"name": name, "has_model": r["model"] is not None, "created": r["created"]}
            for name, r in rooms.items()
        ]
    })


# ---- Instructions Endpoint ----

@app.route('/api/instructions', methods=['GET'])
def get_instructions():
    """Return API documentation and graph JSON schema for AI agents"""

    # Load component/atomic/model indexes to include available types
    base = Path(__file__).parent
    atomics_index = {}
    components_index = {}
    models_index = {}
    atomic_defs = {}
    try:
        with open(base / "atomics" / "index.json", encoding="utf-8") as f:
            atomics_index = json.load(f)
        # Load all atomic definitions
        for cat_key, cat in atomics_index.get("categories", {}).items():
            for fname in cat.get("files", []):
                try:
                    with open(base / "atomics" / fname, encoding="utf-8") as f:
                        data = json.load(f)
                        # Atomic files are plain arrays of component objects
                        items = data if isinstance(data, list) else data.get("components", [])
                        for comp in items:
                            atomic_defs[f"atomic/{cat_key}/{comp['id']}"] = {
                                "name": comp["name"],
                                "inputs": comp.get("inputs", []),
                                "outputs": comp.get("outputs", []),
                                "properties": comp.get("properties", []),
                            }
                except Exception:
                    pass
    except Exception:
        pass
    try:
        with open(base / "components" / "index.json", encoding="utf-8") as f:
            components_index = json.load(f)
    except Exception:
        pass
    try:
        with open(base / "models" / "index.json", encoding="utf-8") as f:
            models_index = json.load(f)
    except Exception:
        pass

    # Build molecular type list
    molecular_types = {}
    for comp in components_index.get("components", []):
        molecular_types[f"molecular/{comp['id']}"] = {
            "name": comp["name"],
            "description": comp.get("description", ""),
        }

    return jsonify({
        "description": "Rapid Neural Designer — API for real-time model collaboration",
        "endpoints": {
            "POST /api/room": {
                "description": "Create a new collaboration room",
                "request_body": None,
                "response": {"room": "<room-name>"},
            },
            "GET /api/room/<room_name>": {
                "description": "Get room info",
                "response": {"room": "<name>", "has_model": True, "created": 0},
            },
            "GET /api/room/<room_name>/model": {
                "description": "Get the current model graph JSON",
                "response": {"room": "<name>", "model": "<graph-json-or-null>"},
            },
            "PUT /api/room/<room_name>/model": {
                "description": "Update the model graph. Broadcasts to all connected WS clients in the room.",
                "request_body": {"model": "<complete-graph-json>"},
                "response": {"room": "<name>", "status": "updated"},
            },
            "GET /api/rooms": {
                "description": "List all active rooms",
            },
            "GET /api/instructions": {
                "description": "This endpoint. Returns API docs and graph schema.",
            },
        },
        "websocket": {
            "description": "SocketIO connection for real-time updates. Clients join a room and receive model_updated events when the API updates the model.",
            "url": "Connect via SocketIO to the backend URL",
            "events": {
                "join_room (client->server)": {"data": {"room": "<room-name>"}},
                "leave_room (client->server)": {"data": {"room": "<room-name>"}},
                "model_updated (server->client)": {
                    "description": "Fired when PUT /api/room/<name>/model is called",
                    "data": {"model": "<complete-graph-json>"},
                },
            },
        },
        "graph_json_schema": {
            "description": "The model field in PUT /api/room/<name>/model must be a LiteGraph-compatible graph JSON object.",
            "format": {
                "last_node_id": "int — highest node ID used",
                "last_link_id": "int — highest link ID used",
                "nodes": [
                    {
                        "id": "int — unique node ID",
                        "type": "string — node type (e.g. 'atomic/math/matmul', 'molecular/linear')",
                        "pos": [0, 0],
                        "size": [220, 100],
                        "flags": {},
                        "order": "int — execution order",
                        "mode": 0,
                        "inputs": [{"name": "input", "type": "tensor", "link": "int|null — link ID"}],
                        "outputs": [{"name": "output", "type": "tensor", "links": ["int — link IDs"]}],
                        "properties": {"key": "value — node-specific properties"},
                        "widgets_values": ["ordered list matching properties"],
                        "title": "string (optional) — custom display title",
                    }
                ],
                "links": [
                    ["[link_id, origin_node_id, origin_slot, target_node_id, target_slot, type_string]"]
                ],
                "groups": [],
                "config": {},
                "extra": {"info": "optional description"},
                "version": 0.4,
            },
            "example_rnn": {
                "last_node_id": 3,
                "last_link_id": 2,
                "nodes": [
                    {
                        "id": 1, "type": "atomic/data/input_tensor",
                        "pos": [100, 50], "size": [220, 130], "flags": {}, "order": 0, "mode": 0,
                        "inputs": [],
                        "outputs": [{"name": "tensor", "type": "tensor", "links": [1]}],
                        "properties": {"batch_size": 4, "seq_len": 32, "dtype": "int", "vocab_size": 10000},
                        "widgets_values": [4, 32, "int", 10000],
                    },
                    {
                        "id": 2, "type": "molecular/embedding",
                        "pos": [100, 250], "size": [220, 100], "flags": {}, "order": 1, "mode": 0,
                        "inputs": [{"name": "token_ids", "type": "tensor", "link": 1}],
                        "outputs": [{"name": "embeddings", "type": "tensor", "links": [2]}],
                        "properties": {"vocab_size": 10000, "dim": 128, "init_std": 0.02},
                        "widgets_values": [10000, 128, 0.02],
                    },
                    {
                        "id": 3, "type": "molecular/linear",
                        "pos": [100, 420], "size": [220, 130], "flags": {}, "order": 2, "mode": 0,
                        "inputs": [{"name": "input", "type": "tensor", "link": 2}],
                        "outputs": [{"name": "output", "type": "tensor", "links": []}],
                        "properties": {"in_features": 128, "out_features": 10, "use_bias": True, "init_std": 0.02},
                        "widgets_values": [128, 10, True, 0.02],
                    },
                ],
                "links": [
                    [1, 1, 0, 2, 0, "tensor"],
                    [2, 2, 0, 3, 0, "tensor"],
                ],
                "groups": [], "config": {}, "extra": {"info": "Simple embedding -> linear"}, "version": 0.4,
            },
        },
        "available_node_types": {
            "atomics": atomic_defs,
            "molecular": molecular_types,
        },
        "tips": [
            "Node IDs must be unique integers. Increment last_node_id for each new node.",
            "Link format: [link_id, source_node_id, source_slot_index, target_node_id, target_slot_index, 'tensor']",
            "Link IDs must be unique integers. Increment last_link_id for each new link.",
            "The 'link' field on an input slot must match a link_id. The 'links' array on an output slot lists link_ids.",
            "widgets_values must be ordered to match the properties list of the node type.",
            "Use molecular/ types for high-level components (linear, attention, etc.) and atomic/ types for primitives.",
            "Set pos to [x, y] coordinates. Stack nodes vertically with ~170px spacing for readability.",
            "Always send the COMPLETE graph JSON — partial updates are not supported.",
        ],
    })


# ---- SocketIO Events ----

@socketio.on('join_room')
def handle_join(data):
    room = data.get('room')
    if room and room in rooms:
        join_room(room)
        return {"status": "joined", "room": room}
    return {"status": "error", "message": "Room not found"}


@socketio.on('leave_room')
def handle_leave(data):
    room = data.get('room')
    if room:
        leave_room(room)
        return {"status": "left", "room": room}


# ---- Security configuration ----

ALLOWED_IMPORTS = {
    'numpy', 'np', 'torch', 'nn', 'transformers', 'soundfile', 'sf',
    'PIL', 'Image', 'json', 'time', 'typing', 'dataclasses', 'dataclass',
    'field', 'Dict', 'Any', 'List', 'Optional', 'Tuple', 'collections',
    'qwen_omni_utils', 'process_mm_info', 'matplotlib', 'plt', 'pandas', 'pd',
    'scipy', 'sklearn', 'cv2', 'tqdm', 'pathlib', 'Path'
}

MAX_EXECUTION_TIME = 300  # 5 minutes
MAX_OUTPUT_SIZE = 1024 * 1024  # 1MB

def validate_code(code):
    """Basic security validation of code"""
    dangerous_patterns = [
        'os.system', 'subprocess.', 'eval(', 'exec(', '__import__',
        'open(', 'file(', 'input(', 'raw_input(',
        'compile(', 'globals(', 'locals(', 'vars(',
        'rmdir', 'remove', 'unlink', 'delete'
    ]

    for pattern in dangerous_patterns:
        if pattern in code:
            return False, f"Forbidden pattern detected: {pattern}"

    return True, "OK"

def extract_imports(code):
    """Extract import statements from code"""
    imports = []
    for line in code.split('\n'):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            imports.append(stripped)
    return imports

def check_imports(code):
    """Verify all imports are in whitelist"""
    imports = extract_imports(code)
    for imp in imports:
        if imp.startswith('import '):
            module = imp.split()[1].split('.')[0].split(' as ')[0]
        elif imp.startswith('from '):
            module = imp.split()[1].split('.')[0]
        else:
            continue

        if module not in ALLOWED_IMPORTS:
            return False, f"Import not allowed: {module}"

    return True, "OK"


# ---- Existing Endpoints ----

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Backend is running'})

@app.route('/execute', methods=['POST'])
def execute_code():
    """Execute Python code in sandboxed environment"""
    try:
        data = request.json
        code = data.get('code', '')

        if not code.strip():
            return jsonify({
                'success': False,
                'error': 'No code provided'
            }), 400

        valid, msg = validate_code(code)
        if not valid:
            return jsonify({
                'success': False,
                'error': f'Security check failed: {msg}'
            }), 403

        valid, msg = check_imports(code)
        if not valid:
            return jsonify({
                'success': False,
                'error': f'Import check failed: {msg}'
            }), 403

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=MAX_EXECUTION_TIME,
                cwd=os.path.dirname(temp_file)
            )
            execution_time = time.time() - start_time

            stdout = result.stdout[:MAX_OUTPUT_SIZE]
            stderr = result.stderr[:MAX_OUTPUT_SIZE]

            return jsonify({
                'success': result.returncode == 0,
                'stdout': stdout,
                'stderr': stderr,
                'returncode': result.returncode,
                'execution_time': round(execution_time, 2)
            })

        except subprocess.TimeoutExpired:
            return jsonify({
                'success': False,
                'error': f'Execution timeout ({MAX_EXECUTION_TIME}s)',
                'stdout': '',
                'stderr': ''
            }), 408

        finally:
            try:
                os.unlink(temp_file)
            except:
                pass

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/validate', methods=['POST'])
def validate_code_endpoint():
    """Validate code without executing"""
    try:
        data = request.json
        code = data.get('code', '')

        valid, msg = validate_code(code)
        if not valid:
            return jsonify({
                'valid': False,
                'error': msg
            })

        valid, msg = check_imports(code)
        if not valid:
            return jsonify({
                'valid': False,
                'error': msg
            })

        return jsonify({
            'valid': True,
            'message': 'Code validation passed'
        })

    except Exception as e:
        return jsonify({
            'valid': False,
            'error': f'Validation error: {str(e)}'
        }), 500

@app.route('/validate-xml', methods=['POST'])
def validate_xml():
    """Load XML file, generate code, and execute to catch runtime errors"""
    try:
        data = request.json
        xml_path = data.get('xml_path', '')

        if not xml_path:
            return jsonify({
                'success': False,
                'error': 'No xml_path provided'
            }), 400

        xml_file = Path(xml_path)
        if not xml_file.exists():
            return jsonify({
                'success': False,
                'error': f'XML file not found: {xml_path}'
            }), 404

        with open(xml_file, 'r', encoding='utf-8') as f:
            xml_content = f.read()

        return jsonify({
            'success': False,
            'error': 'XML validation requires code generation - please send generated code via /execute endpoint'
        }), 501

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Validation error: {str(e)}'
        }), 500


# ======================================================================
# RND Platform API (SDD §9.1)
# ======================================================================

# ---- Team ----

@app.route('/api/rnd/team', methods=['GET'])
def rnd_get_team():
    teams = rnd_repo.list_entities("team")
    if not teams:
        return api_error("No team configured", 404)
    return jsonify(teams[0].to_dict())


# ---- Programs ----

@app.route('/api/rnd/programs', methods=['GET'])
def rnd_list_programs():
    return jsonify([p.to_dict() for p in rnd_repo.list_entities("program")])

@app.route('/api/rnd/programs', methods=['POST'])
def rnd_create_program():
    data = request.get_json()
    if not data or "name" not in data:
        return api_error("'name' required")
    teams = rnd_repo.list_entities("team")
    if not teams:
        return api_error("No team configured", 500)
    prog = Program.create(team_id=teams[0].id, name=data["name"],
                          description=data.get("description", ""))
    rnd_repo.save(prog)
    return jsonify(prog.to_dict()), 201

@app.route('/api/rnd/programs/<program_id>', methods=['GET'])
def rnd_get_program(program_id):
    prog = rnd_repo.load("program", program_id)
    if not prog:
        return api_error("Program not found", 404)
    return jsonify(prog.to_dict())


# ---- Projects ----

@app.route('/api/rnd/projects', methods=['GET'])
def rnd_list_projects():
    program_id = request.args.get("program_id")
    projects = rnd_repo.list_entities("project")
    if program_id:
        projects = [p for p in projects if p.program_id == program_id]
    return jsonify([p.to_dict() for p in projects])

@app.route('/api/rnd/projects', methods=['POST'])
def rnd_create_project():
    data = request.get_json()
    if not data or "name" not in data or "program_id" not in data:
        return api_error("'name' and 'program_id' required")
    proj = Project.create(program_id=data["program_id"], name=data["name"],
                          description=data.get("description", ""))
    rnd_repo.save(proj)
    return jsonify(proj.to_dict()), 201

@app.route('/api/rnd/projects/<project_id>', methods=['GET'])
def rnd_get_project(project_id):
    proj = rnd_repo.load("project", project_id)
    if not proj:
        return api_error("Project not found", 404)
    return jsonify(proj.to_dict())


# ---- Threads ----

@app.route('/api/rnd/threads', methods=['GET'])
def rnd_list_threads():
    project_id = request.args.get("project_id")
    threads = rnd_repo.list_entities("thread")
    if project_id:
        threads = [t for t in threads if t.project_id == project_id]
    return jsonify([t.to_dict() for t in threads])

@app.route('/api/rnd/threads', methods=['POST'])
def rnd_create_thread():
    data = request.get_json()
    if not data or "question" not in data or "project_id" not in data:
        return api_error("'question' and 'project_id' required")
    thread = Thread.create(project_id=data["project_id"], question=data["question"],
                           resolution_criterion=data.get("resolution_criterion", ""))
    rnd_repo.save(thread)
    return jsonify(thread.to_dict()), 201

@app.route('/api/rnd/threads/<thread_id>', methods=['GET'])
def rnd_get_thread(thread_id):
    thread = rnd_repo.load("thread", thread_id)
    if not thread:
        return api_error("Thread not found", 404)
    return jsonify(thread.to_dict())

@app.route('/api/rnd/threads/<thread_id>', methods=['PATCH'])
def rnd_update_thread(thread_id):
    thread = rnd_repo.load("thread", thread_id)
    if not thread:
        return api_error("Thread not found", 404)
    data = request.get_json()
    if "state" in data:
        thread.state = ThreadState(data["state"])
    if "question" in data:
        thread.question = data["question"]
    if "resolution_criterion" in data:
        thread.resolution_criterion = data["resolution_criterion"]
    thread.updated_at = now_iso()
    rnd_repo.save(thread)
    return jsonify(thread.to_dict())


# ---- Statements ----

@app.route('/api/rnd/statements', methods=['GET'])
def rnd_list_statements():
    thread_id = request.args.get("thread_id")
    stmts = rnd_repo.list_entities("statement")
    if thread_id:
        stmts = [s for s in stmts if s.thread_id == thread_id]
    return jsonify([s.to_dict() for s in stmts])

@app.route('/api/rnd/statements', methods=['POST'])
def rnd_create_statement():
    data = request.get_json()
    if not data or "hypothesis" not in data or "thread_id" not in data:
        return api_error("'hypothesis' and 'thread_id' required")
    stmt = Statement.create(thread_id=data["thread_id"], hypothesis=data["hypothesis"])
    rnd_repo.save(stmt)
    return jsonify(stmt.to_dict()), 201

@app.route('/api/rnd/statements/<statement_id>', methods=['GET'])
def rnd_get_statement(statement_id):
    stmt = rnd_repo.load("statement", statement_id)
    if not stmt:
        return api_error("Statement not found", 404)
    return jsonify(stmt.to_dict())


# ---- Experiments ----

@app.route('/api/rnd/experiments', methods=['GET'])
def rnd_list_experiments():
    thread_id = request.args.get("thread_id")
    exps = rnd_repo.list_entities("experiment")
    if thread_id:
        exps = [e for e in exps if e.thread_id == thread_id]
    return jsonify([e.to_dict() for e in exps])

@app.route('/api/rnd/experiments', methods=['POST'])
def rnd_create_experiment():
    data = request.get_json()
    if not data or "thread_id" not in data or "inputs" not in data:
        return api_error("'thread_id' and 'inputs' required")
    inputs = ExperimentInputs.from_dict(data["inputs"])
    exp = Experiment.create(
        thread_id=data["thread_id"],
        created_by=data.get("created_by", "user-bake"),
        inputs=inputs,
        hypothesis=data.get("hypothesis", ""),
        expected=data.get("expected", ""),
        method=data.get("method", {}),
        imported=data.get("imported", False),
    )
    if data.get("status"):
        exp.status = ExperimentStatus(data["status"])
    if data.get("observed"):
        exp.observed = ObservedResults.from_dict(data["observed"])
    if data.get("interpretation"):
        exp.interpretation = data["interpretation"]
    rnd_repo.save(exp)
    return jsonify(exp.to_dict()), 201

@app.route('/api/rnd/experiments/<experiment_id>', methods=['GET'])
def rnd_get_experiment(experiment_id):
    exp = rnd_repo.load("experiment", experiment_id)
    if not exp:
        return api_error("Experiment not found", 404)
    return jsonify(exp.to_dict())

@app.route('/api/rnd/experiments/<experiment_id>', methods=['PATCH'])
def rnd_update_experiment(experiment_id):
    exp = rnd_repo.load("experiment", experiment_id)
    if not exp:
        return api_error("Experiment not found", 404)
    data = request.get_json()
    if "status" in data:
        exp.status = ExperimentStatus(data["status"])
    if "observed" in data:
        exp.observed = ObservedResults.from_dict(data["observed"])
    if "interpretation" in data:
        exp.interpretation = data["interpretation"]
    if "evidence" in data:
        exp.evidence = [EvidenceLink.from_dict(e) for e in data["evidence"]]
    if "architecture_id" in data:
        arch = rnd_repo.load("architecture", data["architecture_id"])
        if not arch:
            return api_error(f"Architecture {data['architecture_id']} not found", 404)
        exp.inputs.architecture_ref = arch.id
        exp.inputs.architecture_hash = arch.content_hash
    exp.updated_at = now_iso()
    rnd_repo.save(exp)
    return jsonify(exp.to_dict())

@app.route('/api/rnd/experiments/<experiment_id>/evidence', methods=['POST'])
def rnd_attach_evidence(experiment_id):
    exp = rnd_repo.load("experiment", experiment_id)
    if not exp:
        return api_error("Experiment not found", 404)
    data = request.get_json()
    if not data or "statement_id" not in data or "sign" not in data:
        return api_error("'statement_id' and 'sign' required")
    link = EvidenceLink(
        statement_id=data["statement_id"],
        sign=EvidenceSign(data["sign"]),
        strength=EvidenceStrength(data.get("strength", "moderate")),
        note=data.get("note", ""),
    )
    exp.evidence.append(link)
    exp.updated_at = now_iso()
    rnd_repo.save(exp)
    return jsonify(link.to_dict()), 201


# ---- Findings ----

@app.route('/api/rnd/findings', methods=['GET'])
def rnd_list_findings():
    thread_id = request.args.get("thread_id")
    findings = rnd_repo.list_entities("finding")
    if thread_id:
        findings = [f for f in findings if f.thread_id == thread_id]
    return jsonify([f.to_dict() for f in findings])

@app.route('/api/rnd/findings', methods=['POST'])
def rnd_create_finding():
    data = request.get_json()
    if not data or "thread_id" not in data or "summary" not in data:
        return api_error("'thread_id' and 'summary' required")
    resolutions = []
    for r in data.get("statement_resolutions", []):
        resolutions.append(StatementResolution(
            statement_id=r["statement_id"],
            resolution=FindingResolution(r["resolution"]),
            note=r.get("note", ""),
        ))
    finding = Finding.create(
        thread_id=data["thread_id"],
        summary=data["summary"],
        reasoning=data.get("reasoning", ""),
        statement_resolutions=resolutions,
        experiment_refs=data.get("experiment_refs", []),
    )
    rnd_repo.save(finding)
    if data.get("resolve_thread", False):
        thread = rnd_repo.load("thread", data["thread_id"])
        if thread:
            thread.state = ThreadState.RESOLVED
            thread.updated_at = now_iso()
            rnd_repo.save(thread)
    return jsonify(finding.to_dict()), 201

@app.route('/api/rnd/findings/<finding_id>', methods=['GET'])
def rnd_get_finding(finding_id):
    finding = rnd_repo.load("finding", finding_id)
    if not finding:
        return api_error("Finding not found", 404)
    return jsonify(finding.to_dict())

@app.route('/api/rnd/findings/<finding_id>', methods=['PATCH'])
def rnd_update_finding(finding_id):
    finding = rnd_repo.load("finding", finding_id)
    if not finding:
        return api_error("Finding not found", 404)
    data = request.get_json()
    for field in ["summary", "reasoning", "experiment_refs"]:
        if field in data:
            setattr(finding, field, data[field])
    if "statement_resolutions" in data:
        finding.statement_resolutions = [StatementResolution(
            statement_id=r["statement_id"],
            resolution=FindingResolution(r["resolution"]),
            note=r.get("note", ""),
        ) for r in data["statement_resolutions"]]
    finding.updated_at = now_iso()
    rnd_repo.save(finding)
    return jsonify(finding.to_dict())


# ---- Architectures ----

@app.route('/api/rnd/architectures', methods=['GET'])
def rnd_list_architectures():
    archs = rnd_repo.list_entities("architecture")
    include_archived = request.args.get("include_archived", "false").lower() == "true"
    if not include_archived:
        archs = [a for a in archs if not a.archived]
    return jsonify([a.to_dict() for a in archs])

@app.route('/api/rnd/architectures', methods=['POST'])
def rnd_import_architecture():
    data = request.get_json()
    if not data or "content" not in data:
        return api_error("'content' required")
    arch = rnd_repo.import_architecture(
        name=data.get("name", "unnamed"),
        content=data["content"],
        variant_of=data.get("variant_of"),
    )
    return jsonify(arch.to_dict()), 201

@app.route('/api/rnd/architectures/<architecture_id>', methods=['GET'])
def rnd_get_architecture(architecture_id):
    arch = rnd_repo.load("architecture", architecture_id)
    if not arch:
        return api_error("Architecture not found", 404)
    return jsonify(arch.to_dict())

@app.route('/api/rnd/architectures/<architecture_id>/content', methods=['PUT'])
def rnd_replace_architecture_content(architecture_id):
    arch = rnd_repo.load("architecture", architecture_id)
    if not arch:
        return api_error("Architecture not found", 404)
    data = request.get_json()
    if not data or "content" not in data:
        return api_error("'content' required")
    arch.content = data["content"]
    arch.content_hash = content_hash(arch.content)
    arch.updated_at = now_iso()
    rnd_repo.save(arch)
    return jsonify(arch.to_dict())

@app.route('/api/rnd/architectures/<architecture_id>/archive', methods=['POST'])
def rnd_archive_architecture(architecture_id):
    if rnd_repo.archive("architecture", architecture_id):
        return jsonify({"archived": True})
    return api_error("Architecture not found", 404)

@app.route('/api/rnd/architectures/by-hash/<path:hash_value>', methods=['GET'])
def rnd_get_architecture_by_hash(hash_value):
    arch = rnd_repo.find_architecture_by_hash(hash_value)
    if not arch:
        return api_error("Architecture not found for hash", 404)
    return jsonify(arch.to_dict())


# ---- Papers ----

@app.route('/api/rnd/papers', methods=['GET'])
def rnd_list_papers():
    return jsonify([p.to_dict() for p in rnd_repo.list_entities("paper")])

@app.route('/api/rnd/papers', methods=['POST'])
def rnd_create_paper():
    data = request.get_json()
    if not data or "title" not in data:
        return api_error("'title' required")
    paper = Paper.create(title=data["title"], authors=data.get("authors", []),
                         program_id=data.get("program_id", ""),
                         target=data.get("target", "arxiv"))
    if data.get("project_ids"):
        paper.project_ids = data["project_ids"]
    if data.get("thread_ids"):
        paper.thread_ids = data["thread_ids"]
    rnd_repo.save(paper)
    return jsonify(paper.to_dict()), 201

@app.route('/api/rnd/papers/<paper_id>', methods=['GET'])
def rnd_get_paper(paper_id):
    paper = rnd_repo.load("paper", paper_id)
    if not paper:
        return api_error("Paper not found", 404)
    return jsonify(paper.to_dict())

@app.route('/api/rnd/papers/<paper_id>', methods=['PATCH'])
def rnd_update_paper(paper_id):
    paper = rnd_repo.load("paper", paper_id)
    if not paper:
        return api_error("Paper not found", 404)
    data = request.get_json()
    for field in ["title", "authors", "thread_ids", "project_ids"]:
        if field in data:
            setattr(paper, field, data[field])
    if "status" in data:
        paper.status = PaperStatus(data["status"])
    if "metadata" in data:
        paper.metadata.update(data["metadata"])
    paper.updated_at = now_iso()
    rnd_repo.save(paper)
    return jsonify(paper.to_dict())


# ---- Paper Sections ----

@app.route('/api/rnd/papers/<paper_id>/publish', methods=['POST'])
def rnd_publish_paper(paper_id):
    paper = rnd_repo.load("paper", paper_id)
    if not paper:
        return api_error("Paper not found", 404)
    slug = paper.metadata.get("public_slug")
    if not slug:
        base = re.sub(r'[^a-z0-9]+', '-', paper.title.lower()).strip('-')[:60]
        slug = f"{base}-{secrets.token_hex(3)}"
        paper.metadata["public_slug"] = slug
        paper.metadata["published_at"] = now_iso()
        paper.updated_at = now_iso()
        rnd_repo.save(paper)
    return jsonify({"public_slug": slug, "url": f"/pub/{slug}"})

@app.route('/api/rnd/papers/<paper_id>/publish', methods=['DELETE'])
def rnd_unpublish_paper(paper_id):
    paper = rnd_repo.load("paper", paper_id)
    if not paper:
        return api_error("Paper not found", 404)
    paper.metadata.pop("public_slug", None)
    paper.metadata.pop("published_at", None)
    paper.updated_at = now_iso()
    rnd_repo.save(paper)
    return jsonify({"unpublished": True})

@app.route('/pub/<slug>', methods=['GET'])
def rnd_public_paper(slug):
    # Public, read-only. Only entities explicitly published (metadata.public_slug) are reachable.
    for p in rnd_repo.list_entities("paper"):
        if p.metadata.get("public_slug") == slug:
            sections = []
            for s in p.sections:
                try:
                    content = rnd_repo.load_paper_section_content(s) or ""
                except Exception:
                    content = ""
                # strip internal binding comments from the public view
                content = re.sub(r'<!--.*?-->', '', content, flags=re.S)
                sections.append({"title": s.title, "type": s.section_type.value, "content": content})
            html = render_template_string(PUBLIC_PAPER_TEMPLATE, paper=p, sections=sections)
            return html, 200, {"Content-Type": "text/html; charset=utf-8"}
    return "Not found", 404

PUBLIC_PAPER_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ paper.title }}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/marked@12/marked.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<style>
 body{font-family:Georgia,'Times New Roman',serif;max-width:760px;margin:0 auto;padding:2rem 1.2rem;color:#1a1a1a;line-height:1.65;background:#fdfdfa}
 h1{font-size:1.7rem;line-height:1.3;margin-bottom:.3rem} .authors{color:#555;margin-bottom:.2rem}
 .meta{color:#888;font-size:.85rem;margin-bottom:2rem} h2{font-size:1.25rem;margin-top:2.2rem;border-bottom:1px solid #ddd;padding-bottom:.2rem}
 table{border-collapse:collapse;margin:1rem 0;font-size:.92rem} th,td{border:1px solid #ccc;padding:.35rem .6rem}
 code{background:#f0efe8;padding:.1rem .3rem;border-radius:3px;font-size:.9em}
 .abstract{background:#f5f4ec;padding:1rem 1.3rem;border-left:3px solid #999;font-size:.95rem}
</style></head><body>
<h1>{{ paper.title }}</h1>
<div class="authors">{{ paper.authors | join(', ') }}</div>
<div class="meta">Published {{ paper.metadata.get('published_at','')[:10] }} · Rapid Neural Designer research platform</div>
{% for s in sections %}
<section class="{{ 'abstract' if s.type == 'abstract' else '' }}">
{% if s.type != 'abstract' %}<h2>{{ s.title }}</h2>{% endif %}
<div class="md" data-md="{{ s.content | e }}"></div>
</section>
{% endfor %}
<script>
document.querySelectorAll('.md').forEach(el => { el.innerHTML = marked.parse(el.dataset.md); });
document.addEventListener('DOMContentLoaded', () => renderMathInElement(document.body, {
  delimiters: [{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}], throwOnError: false }));
</script></body></html>"""

@app.route('/api/rnd/papers/<paper_id>/sections', methods=['POST'])
def rnd_add_paper_section(paper_id):
    paper = rnd_repo.load("paper", paper_id)
    if not paper:
        return api_error("Paper not found", 404)
    data = request.get_json()
    if not data or "title" not in data:
        return api_error("'title' required")
    section_id = generate_id("sec")
    section = PaperSection(
        id=section_id,
        section_type=SectionType(data.get("type", "other")),
        title=data["title"],
        content_ref=f"papers/{paper_id}/sections/{section_id}.md",
        bindings=[],
    )
    paper.sections.append(section)
    paper.updated_at = now_iso()
    rnd_repo.save(paper)
    # Write initial content
    content = data.get("content", "")
    rnd_repo.save_paper_section_content(paper, section, content)
    return jsonify(section.to_dict()), 201

@app.route('/api/rnd/papers/<paper_id>/sections/<section_id>/content', methods=['GET'])
def rnd_get_section_content(paper_id, section_id):
    paper = rnd_repo.load("paper", paper_id)
    if not paper:
        return api_error("Paper not found", 404)
    section = next((s for s in paper.sections if s.id == section_id), None)
    if not section:
        return api_error("Section not found", 404)
    content = rnd_repo.load_paper_section_content(section)
    return content, 200, {"Content-Type": "text/markdown; charset=utf-8"}

@app.route('/api/rnd/papers/<paper_id>/sections/<section_id>/content', methods=['PUT'])
def rnd_update_section_content(paper_id, section_id):
    paper = rnd_repo.load("paper", paper_id)
    if not paper:
        return api_error("Paper not found", 404)
    section = next((s for s in paper.sections if s.id == section_id), None)
    if not section:
        return api_error("Section not found", 404)
    content = request.get_data(as_text=True)
    rnd_repo.save_paper_section_content(paper, section, content)
    return jsonify({"updated": True})

@app.route('/api/rnd/papers/<paper_id>/sections/<section_id>', methods=['DELETE'])
def rnd_delete_section(paper_id, section_id):
    paper = rnd_repo.load("paper", paper_id)
    if not paper:
        return api_error("Paper not found", 404)
    paper.sections = [s for s in paper.sections if s.id != section_id]
    paper.updated_at = now_iso()
    rnd_repo.save(paper)
    return jsonify({"deleted": True})


# ---- Generic Citation Attachment ----

@app.route('/api/rnd/cite/<entity_id>', methods=['POST'])
def rnd_add_citation(entity_id):
    """Attach a disclosure or external citation to any entity that supports citations."""
    prefix = entity_id.split("-")[0] if "-" in entity_id else ""
    prefix_to_type = {
        "thread": "thread", "stmt": "statement", "exp": "experiment",
        "find": "finding", "paper": "paper",
    }
    entity_type = prefix_to_type.get(prefix)
    if not entity_type:
        return api_error(f"Entity type '{prefix}' does not support citations")
    entity = rnd_repo.load(entity_type, entity_id)
    if not entity:
        return api_error("Entity not found", 404)
    if not hasattr(entity, 'citations'):
        return api_error("Entity does not support citations")
    data = request.get_json()
    cit = Citation.from_dict(data)
    entity.citations.append(cit)
    entity.updated_at = now_iso()
    rnd_repo.save(entity)
    return jsonify(cit.to_dict()), 201

@app.route('/api/rnd/cite/<entity_id>', methods=['GET'])
def rnd_get_citations(entity_id):
    prefix = entity_id.split("-")[0] if "-" in entity_id else ""
    prefix_to_type = {
        "thread": "thread", "stmt": "statement", "exp": "experiment",
        "find": "finding", "paper": "paper",
    }
    entity_type = prefix_to_type.get(prefix)
    if not entity_type:
        return api_error(f"Entity type does not support citations")
    entity = rnd_repo.load(entity_type, entity_id)
    if not entity:
        return api_error("Entity not found", 404)
    return jsonify([c.to_dict() for c in getattr(entity, 'citations', [])])


# ---- Disclosures ----

@app.route('/api/rnd/disclosures', methods=['GET'])
def rnd_list_disclosures():
    return jsonify([d.to_dict() for d in rnd_repo.list_entities("disclosure")])

@app.route('/api/rnd/disclosures', methods=['POST'])
def rnd_create_disclosure():
    data = request.get_json()
    if not data or "title" not in data:
        return api_error("'title' required")
    disc = Disclosure.create(
        title=data["title"],
        disclosure_type=DisclosureType(data.get("type", "other")),
        created_by=data.get("created_by", "user-bake"),
        tags=data.get("tags", []),
    )
    if data.get("source_url"):
        disc.source_url = data["source_url"]
    if data.get("visibility"):
        disc.visibility = Visibility(data["visibility"])
    rnd_repo.save(disc)
    if data.get("content"):
        rnd_repo.save_disclosure_content(disc, data["content"])
    return jsonify(disc.to_dict()), 201

@app.route('/api/rnd/disclosures/<disclosure_id>', methods=['GET'])
def rnd_get_disclosure(disclosure_id):
    disc = rnd_repo.load("disclosure", disclosure_id)
    if not disc:
        return api_error("Disclosure not found", 404)
    result = disc.to_dict()
    if request.args.get("include_content", "false").lower() == "true":
        result["content"] = rnd_repo.load_disclosure_content(disc)
    return jsonify(result)

@app.route('/api/rnd/disclosures/<disclosure_id>/content', methods=['GET'])
def rnd_get_disclosure_content(disclosure_id):
    disc = rnd_repo.load("disclosure", disclosure_id)
    if not disc:
        return api_error("Disclosure not found", 404)
    content = rnd_repo.load_disclosure_content(disc)
    return content, 200, {"Content-Type": "text/markdown; charset=utf-8"}

@app.route('/api/rnd/disclosures/<disclosure_id>/content', methods=['PUT'])
def rnd_update_disclosure_content(disclosure_id):
    disc = rnd_repo.load("disclosure", disclosure_id)
    if not disc:
        return api_error("Disclosure not found", 404)
    content = request.get_data(as_text=True)
    rnd_repo.save_disclosure_content(disc, content)
    return jsonify({"updated": True})


# ---- Search & Graph Queries ----

@app.route('/api/rnd/search', methods=['GET'])
def rnd_search():
    q = request.args.get("q", "")
    entity_type = request.args.get("type")
    if not q:
        return api_error("'q' query parameter required")
    ensure_index()
    results = rnd_index.search(q, entity_type=entity_type)
    return jsonify(results)

@app.route('/api/rnd/graph/children/<entity_id>', methods=['GET'])
def rnd_graph_children(entity_id):
    ensure_index()
    return jsonify(rnd_index.find_children(entity_id))

@app.route('/api/rnd/graph/evidence/<statement_id>', methods=['GET'])
def rnd_graph_evidence(statement_id):
    ensure_index()
    return jsonify(rnd_index.get_evidence_for_statement(statement_id))

@app.route('/api/rnd/graph/citations/<entity_id>', methods=['GET'])
def rnd_graph_citations(entity_id):
    ensure_index()
    return jsonify(rnd_index.get_citations_for_entity(entity_id))

@app.route('/api/rnd/graph/backlinks/<disclosure_id>', methods=['GET'])
def rnd_graph_backlinks(disclosure_id):
    ensure_index()
    return jsonify(rnd_index.get_disclosure_backlinks(disclosure_id))

@app.route('/api/rnd/graph/variants/<architecture_id>', methods=['GET'])
def rnd_graph_variants(architecture_id):
    ensure_index()
    return jsonify(rnd_index.get_architecture_variants(architecture_id))

@app.route('/api/rnd/index/rebuild', methods=['POST'])
def rnd_rebuild_index():
    ensure_index()
    stats = rnd_index.rebuild(rnd_repo.root)
    return jsonify(stats)


# ------------------------------------------------------------------
# User Components
# ------------------------------------------------------------------

@app.route('/api/rnd/user-components', methods=['GET'])
def rnd_list_user_components():
    user = get_current_user()
    if not user:
        return jsonify([])
    rows = auth_db.list_user_components(user['id'])
    for r in rows:
        r['definition'] = json.loads(r['definition'])
    return jsonify(rows)


@app.route('/api/rnd/user-components', methods=['POST'])
def rnd_create_user_component():
    data = request.get_json()
    if not data or 'definition' not in data or 'kind' not in data:
        return api_error("'kind' and 'definition' required")
    user = request.user
    result = auth_db.create_user_component(
        owner_id=user['id'],
        kind=data['kind'],
        category=data.get('category', ''),
        definition=json.dumps(data['definition']) if isinstance(data['definition'], dict) else data['definition'],
    )
    result['definition'] = json.loads(result['definition']) if isinstance(result['definition'], str) else result['definition']
    return jsonify(result), 201


@app.route('/api/rnd/user-components/<comp_id>', methods=['DELETE'])
def rnd_delete_user_component(comp_id):
    user = request.user
    if not auth_db.delete_user_component(comp_id, user['id']):
        return api_error("Component not found or not owned by you", 404)
    return jsonify({"deleted": True})


@app.route('/api/rnd/user-components/<comp_id>/promote', methods=['POST'])
def rnd_promote_user_component(comp_id):
    user = request.user
    if not auth_db.is_admin(user['id']):
        return api_error("Admin only", 403)
    uc = auth_db.get_user_component(comp_id)
    if not uc:
        return api_error("Component not found", 404)

    from rnd.component_catalog import ComponentCatalog
    catalog = ComponentCatalog(STATIC_DIR)
    definition = json.loads(uc['definition'])

    if uc['kind'] == 'component':
        catalog.promote_component(definition['id'], definition)
    elif uc['kind'] == 'atomic':
        category = uc.get('category') or definition.get('category', '')
        if not category:
            return api_error("Atomic must have a category")
        catalog.promote_atomic(definition, category)
    else:
        return api_error(f"Unknown kind: {uc['kind']}")

    # Remove from user_components
    auth_db.delete_user_component(comp_id, uc['owner_id'])
    return jsonify({"promoted": True, "kind": uc['kind'], "id": definition['id']})


# ------------------------------------------------------------------
# Admin: Deploy (git pull + restart)
# ------------------------------------------------------------------

@app.route('/api/rnd/admin/deploy', methods=['POST'])
def admin_deploy():
    user = get_current_user()
    if not user or not auth_db.is_admin(user['id']):
        return api_error("Admin only", 403)

    import subprocess as _sp

    # Git pull
    result = _sp.run(
        ['git', 'pull', 'origin', 'master'],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
    )
    pull_output = result.stdout.strip() + '\n' + result.stderr.strip()

    if result.returncode != 0:
        return jsonify({"ok": False, "phase": "git pull", "output": pull_output}), 500

    # Schedule restart after response is sent
    _sp.Popen(
        'sleep 1 && systemctl restart rnd 2>/dev/null || true',
        shell=True, start_new_session=True,
    )

    return jsonify({"ok": True, "output": pull_output, "restarting": True})


# ======================================================================
# Server startup
# ======================================================================

if __name__ == '__main__':
    init_platform()
    print("=" * 60)
    print("RND Platform — Unified Backend")
    print("=" * 60)
    print(f"  Editor:   http://localhost:5000  (rooms, execute, validate)")
    print(f"  Platform: http://localhost:5000/api/rnd/*")
    print(f"  MCP:      http://localhost:5000/mcp")
    print(f"  Repo:     {rnd_repo.root}")
    print(f"  Max exec: {MAX_EXECUTION_TIME}s")
    print("=" * 60)
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
