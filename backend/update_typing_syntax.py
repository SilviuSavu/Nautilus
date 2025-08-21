#!/usr/bin/env python3
"""
Script to update Python typing syntax to Python 3.13 modern syntax
"""

import os
import re
from pathlib import Path

def update_file_typing(file_path: Path) -> bool:
    """Update a single file's typing syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Update typing imports - remove Union, Optional, Dict, List, etc.
        content = re.sub(
            r'from typing import ([^,\n]*,\s*)*(Union|Optional|Dict|List|Tuple)([,\s][^,\n]*)*',
            lambda m: 'from typing import ' + ', '.join([
                item.strip() for item in m.group(0).replace('from typing import ', '').split(',')
                if item.strip() and item.strip() not in ['Union', 'Optional', 'Dict', 'List', 'Tuple']
            ]) if any(item.strip() not in ['Union', 'Optional', 'Dict', 'List', 'Tuple'] 
                     for item in m.group(0).replace('from typing import ', '').split(',') 
                     if item.strip()) else '',
            content
        )
        
        # Clean up empty typing imports
        content = re.sub(r'from typing import\s*\n', '', content)
        content = re.sub(r'from typing import\s*$', '', content)
        
        # Update Optional[X] to X | None
        content = re.sub(r'Optional\[([^\]]+)\]', r'\1 | None', content)
        
        # Update Union[X, Y] to X | Y
        content = re.sub(r'Union\[([^\]]+)\]', lambda m: ' | '.join(
            item.strip() for item in m.group(1).split(',')
        ), content)
        
        # Update dict[X, Y] to dict[X, Y]
        content = re.sub(r'Dict\[([^\]]+)\]', r'dict[\1]', content)
        
        # Update list[X] to list[X]
        content = re.sub(r'List\[([^\]]+)\]', r'list[\1]', content)
        
        # Update Tuple[X, Y] to tuple[X, Y]
        content = re.sub(r'Tuple\[([^\]]+)\]', r'tuple[\1]', content)
        
        # If content changed, write it back
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def main():
    """Update all Python files in the backend directory"""
    backend_dir = Path(__file__).parent
    updated_files = []
    
    for py_file in backend_dir.rglob('*.py'):
        if py_file.name == 'update_typing_syntax.py':
            continue
            
        if update_file_typing(py_file):
            updated_files.append(py_file.name)
            print(f"✓ Updated {py_file.name}")
    
    if updated_files:
        print(f"\n✓ Updated {len(updated_files)} files with Python 3.13 typing syntax:")
        for file in updated_files:
            print(f"  - {file}")
    else:
        print("No files needed updating")

if __name__ == "__main__":
    main()