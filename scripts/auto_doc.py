import os
import ast

def add_docstrings(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') and file != 'auto_doc.py':
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    tree = ast.parse(content)
                except Exception:
                    continue
                
                lines = content.split('\n')
                inserts = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Check if it already has a docstring
                        if ast.get_docstring(node) is None:
                            # We need to insert a docstring right after the function definition
                            # Function body starts at node.body[0].lineno
                            insert_line = node.body[0].lineno - 1
                            indent = " " * node.body[0].col_offset
                            doc = f'{indent}"""\n{indent}Chi tiết hàm: `{node.name}`\n{indent}- Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.\n{indent}- Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.\n{indent}"""'
                            inserts.append((insert_line, doc))
                            
                if inserts:
                    inserts.sort(key=lambda x: x[0], reverse=True)
                    for line_idx, doc in inserts:
                        lines.insert(line_idx, doc)
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(lines))
                    print(f"Added docstrings to {filepath}")

add_docstrings('blocks')
add_docstrings('data')
add_docstrings('engine')
add_docstrings('layers')
add_docstrings('loss_fn')
add_docstrings('optim')
