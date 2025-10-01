#!/usr/bin/env python3
"""
Simple XML tester - loads XML, sends to backend for execution
"""

import sys
import requests

if len(sys.argv) < 2:
    print("Usage: python test_xml.py <xml_file>")
    sys.exit(1)

xml_file = sys.argv[1]

print(f"Testing: {xml_file}")
print("\nTo test this XML file:")
print("1. Open web_interface/index.html in browser")
print("2. Click '📁 Load' and select the XML file")
print("3. Click '▶️ Run Code'")
print("4. Check output panel for errors")
print("\nThe error you saw was:")
print("  NameError: name 'EmbeddingAtom' is not defined")
print("\nThis means the code generator is creating references to")
print("classes like 'EmbeddingAtom' but not importing them.")