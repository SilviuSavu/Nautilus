#!/usr/bin/env python3
import re
import sys

def analyze_security(file_path):
    """Analyze security vulnerabilities in code"""
    with open(file_path, 'r') as f:
        content = f.read()

    vulnerabilities = []

    # Check for hardcoded secrets/keys
    if re.search(r'password\s*=\s*["\'][\w\d]+["\']', content, re.IGNORECASE):
        vulnerabilities.append('Hardcoded passwords detected')
        
    if re.search(r'api_key\s*=\s*["\'][\w\d]+["\']', content, re.IGNORECASE):
        vulnerabilities.append('Hardcoded API keys detected')

    # Check for SQL injection risks
    if re.search(r'query\s*=\s*["\'].*%s.*["\']', content):
        vulnerabilities.append('Potential SQL injection risk')

    # Check for eval/exec usage
    if re.search(r'\b(eval|exec)\s*\(', content):
        vulnerabilities.append('Use of eval/exec detected')

    # Check for unsafe pickle usage
    if re.search(r'pickle\.loads?\(', content):
        vulnerabilities.append('Unsafe pickle usage detected')

    # Check for shell injection
    if re.search(r'os\.system\(|subprocess\.(call|run|Popen)', content):
        vulnerabilities.append('Potential shell injection risk')

    # Check error handling patterns
    try_blocks = len(re.findall(r'\btry\s*:', content))
    except_blocks = len(re.findall(r'\bexcept\s*:', content))

    print('=== SECURITY ANALYSIS ===')
    if vulnerabilities:
        for vuln in vulnerabilities:
            print(f'❌ {vuln}')
    else:
        print('✅ No obvious security vulnerabilities detected')

    print(f'✅ Error handling: {try_blocks} try blocks, {except_blocks} except blocks')

    # Check for proper validation
    if 'HTTPException' in content:
        print('✅ HTTP exception handling present')
    if 'logging' in content:
        print('✅ Logging implemented')
    if 'timeout' in content.lower():
        print('✅ Timeout handling present')
    
    return len(vulnerabilities) == 0

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "engines/marketdata/centralized_marketdata_hub.py"
    analyze_security(file_path)