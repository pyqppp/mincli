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
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    chunk = f.read(4096)
                if '\x00' in chunk:
                    return f"不支持二进制格式: {ext}"
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception:
                return f"不支持的文件格式: {ext}"

        if not content.strip():
            return f"文件内容为空: {filename}"

        return f"{filename}：\n{content.strip()}"
    except Exception as e:
        return f"文件解析失败: {e}"


def write_file_content(filepath: str, content: str) -> str:
    filepath = os.path.expanduser(filepath)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"已成功写入 {len(content)} 字符到 {filepath}"
    except Exception as e:
        return f"写入失败: {e}"


def edit_file_content(filepath: str, old_string: str, new_string: str) -> str:
    filepath = os.path.expanduser(filepath)
    if not os.path.exists(filepath):
        return f"文件不存在: {filepath}"
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return f"读取文件失败: {e}"

    if old_string not in content:
        return "未找到匹配的原文，请确保 old_string 与文件内容完全一致（包括空格和换行）"

    new_content = content.replace(old_string, new_string, 1)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"已成功替换文件 {filepath}"
    except Exception as e:
        return f"写入失败: {e}"


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
