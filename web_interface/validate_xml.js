#!/usr/bin/env node
/**
 * XML Validation Tool for Neural VM Builder
 * Loads an XML file, generates Python code using Blockly, and executes it
 * to catch runtime errors.
 */

const fs = require('fs');
const { execSync } = require('child_process');
const path = require('path');

// Check if XML file is provided
if (process.argv.length < 3) {
    console.log('Usage: node validate_xml.js <xml_file_path>');
    console.log('Example: node validate_xml.js simple_gpt_example.xml');
    process.exit(1);
}

const xmlPath = process.argv[2];

console.log('='.repeat(60));
console.log('Neural VM Builder - XML Validation Tool');
console.log('='.repeat(60));

// Check if file exists
if (!fs.existsSync(xmlPath)) {
    console.log(`❌ Error: XML file not found: ${xmlPath}`);
    process.exit(1);
}

console.log(`📁 Loading XML: ${xmlPath}`);

// Read XML content
const xmlContent = fs.readFileSync(xmlPath, 'utf8');

console.log('🔄 Generating Python code from XML...\n');

// Generate code by sending POST request to backend
const http = require('http');

const postData = JSON.stringify({ xml_content: xmlContent });

const options = {
    hostname: 'localhost',
    port: 5000,
    path: '/validate-xml-content',
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(postData)
    }
};

const req = http.request(options, (res) => {
    let data = '';

    res.on('data', (chunk) => {
        data += chunk;
    });

    res.on('end', () => {
        try {
            const result = JSON.parse(data);

            if (result.success) {
                console.log('✅ Validation passed!');
                console.log(`\n📊 Execution time: ${result.execution_time}s`);
                if (result.stdout) {
                    console.log(`\n📤 Output:\n${result.stdout}`);
                }
            } else {
                console.log(`❌ Validation failed!`);
                console.log(`\n⚠️  Error: ${result.error}`);
                if (result.stderr) {
                    console.log(`\n📋 Details:\n${result.stderr}`);
                }
            }
        } catch (e) {
            console.log(`❌ Failed to parse response: ${e.message}`);
        }
    });
});

req.on('error', (e) => {
    console.log(`❌ Error: Cannot connect to backend at http://localhost:5000`);
    console.log('Make sure the backend is running: python backend.py');
    process.exit(1);
});

req.write(postData);
req.end();