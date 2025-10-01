#!/usr/bin/env python3
"""
XML Validation Tool for Neural VM Builder
Loads an XML file, sends it to the backend for code generation and execution,
then reports any runtime errors found.
"""

import sys
import requests
import json
from pathlib import Path

BACKEND_URL = "http://localhost:5000"

def load_and_generate_code(xml_path):
    """
    Load XML file and use selenium/playwright to generate Python code.
    For now, we'll just instruct the user to use the web interface.
    """
    print(f"Loading XML from: {xml_path}")

    xml_file = Path(xml_path)
    if not xml_file.exists():
        print(f"❌ Error: XML file not found: {xml_path}")
        return None

    print("\n⚠️  This tool requires the web interface to be open.")
    print("Please:")
    print("1. Open web_interface/index.html in your browser")
    print("2. Load the XML file using the '📁 Load' button")
    print("3. Click the '▶️ Run Code' button")
    print("4. Check the execution output panel for errors")

    return None

def validate_xml_via_api(xml_path):
    """Send XML path to backend for validation"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/validate-xml",
            json={"xml_path": str(xml_path)},
            timeout=10
        )
        return response.json()
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Cannot connect to backend at {BACKEND_URL}")
        print("Make sure the backend is running: python backend.py")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_xml.py <xml_file_path>")
        print("Example: python validate_xml.py simple_gpt_example.xml")
        sys.exit(1)

    xml_path = sys.argv[1]

    print("=" * 60)
    print("Neural VM Builder - XML Validation Tool")
    print("=" * 60)

    # Try API approach first
    result = validate_xml_via_api(xml_path)

    if result and result.get('success'):
        print("✅ Validation passed!")
        print(f"\nOutput:\n{result.get('stdout', '')}")
    elif result:
        print(f"❌ Validation failed: {result.get('error', 'Unknown error')}")
        if result.get('stderr'):
            print(f"\nError details:\n{result.get('stderr')}")
    else:
        # Fall back to manual instructions
        load_and_generate_code(xml_path)

if __name__ == "__main__":
    main()