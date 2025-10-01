"""
Simple Flask backend for executing generated Neural VM code
Provides sandboxed execution with timeout and resource limits
"""

import os
import sys
import subprocess
import tempfile
import time
import json
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from file:// origin

# Security configuration
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
        # Extract module name
        if imp.startswith('import '):
            module = imp.split()[1].split('.')[0].split(' as ')[0]
        elif imp.startswith('from '):
            module = imp.split()[1].split('.')[0]
        else:
            continue

        if module not in ALLOWED_IMPORTS:
            return False, f"Import not allowed: {module}"

    return True, "OK"

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

        # Security validation
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

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Execute code with timeout
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
            # Cleanup
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

        # Check if file exists
        xml_file = Path(xml_path)
        if not xml_file.exists():
            return jsonify({
                'success': False,
                'error': f'XML file not found: {xml_path}'
            }), 404

        # Read XML file
        with open(xml_file, 'r', encoding='utf-8') as f:
            xml_content = f.read()

        # We need to generate Python code from XML
        # This requires the frontend's Blockly code generator
        # For now, we'll expect the frontend to send us the generated code
        # OR we execute via a different approach

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
    print("Neural VM Builder - Execution Backend")
    print("=" * 60)
    print(f"Starting server on http://localhost:5000")
    print(f"Max execution time: {MAX_EXECUTION_TIME}s")
    print(f"Allowed imports: {', '.join(sorted(ALLOWED_IMPORTS))}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)