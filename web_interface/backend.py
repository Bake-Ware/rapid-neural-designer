"""
Simple Flask backend for executing generated Neural VM code
Provides sandboxed execution with timeout and resource limits
Includes room-based collaboration via SocketIO
"""

import os
import sys
import subprocess
import tempfile
import time
import json
import random
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, join_room, leave_room

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

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


if __name__ == '__main__':
    print("=" * 60)
    print("Rapid Neural Designer - Backend")
    print("=" * 60)
    print(f"Starting server on http://localhost:5000")
    print(f"Max execution time: {MAX_EXECUTION_TIME}s")
    print(f"Room system: enabled")
    print(f"WebSocket: enabled (SocketIO)")
    print("=" * 60)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
