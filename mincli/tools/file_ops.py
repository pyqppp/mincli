import os
import csv


def parse_file(filepath: str) -> str:
    filepath = os.path.expanduser(filepath)
    if not os.path.exists(filepath):
        return f"文件不存在: {filepath}"
    ext = os.path.splitext(filepath)[1].lower()
    filename = os.path.basename(filepath)
    content = ""

    try:
        if ext in ('.txt', '.md', '.py', '.bat', '.sh'):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        elif ext == '.csv':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                rows = [','.join(row) for row in reader]
                content = '\n'.join(rows)
        elif ext == '.pdf':
            try:
                from pdfminer.high_level import extract_text
                content = extract_text(filepath)
            except ImportError:
                return "需安装 pdfminer.six: pip install pdfminer.six"
        elif ext == '.docx':
            try:
                from docx import Document
                doc = Document(filepath)
                content = '\n'.join([p.text for p in doc.paragraphs])
            except ImportError:
                return "需安装 python-docx: pip install python-docx"
        elif ext == '.doc':
            return "不支持 .doc 格式，请转换为 .docx 或 .txt"
        else:
            return f"不支持的文件格式: {ext}"

        if not content.strip():
            return f"文件内容为空: {filename}"

        return f"{filename}：\n{content.strip()}"
    except Exception as e:
        return f"文件解析失败: {e}"


def list_directory(directory: str, show_hidden: bool = False) -> str:
    directory = os.path.expanduser(directory)
    if not os.path.isdir(directory):
        return f"目录不存在: {directory}"
    try:
        entries = []
        for entry in os.scandir(directory):
            if not show_hidden and entry.name.startswith("."):
                continue
            prefix = "[目录] " if entry.is_dir() else "[文件] "
            entries.append(f"{prefix}{entry.name}")
        if not entries:
            return "(空目录)"
        return f"目录: {directory}\n" + "\n".join(entries)
    except PermissionError:
        return f"(权限不足，无法读取 {directory})"
    except Exception as e:
        return f"读取目录失败: {e}"
