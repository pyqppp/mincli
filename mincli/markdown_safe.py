"""markdown_it 防御补丁。

背景：markdown-it-py（Textual Markdown 组件底层解析器）在解析某些极端输入
（例如流式增量把「引用块内的表格」截断成不完整结构、且文档以空行结尾）时，
html_block / table 等块级规则会越界访问 state.src[pos] 抛 IndexError，
导致整个 TUI 崩溃（此前已在真实使用中复现）。

本模块包装 MarkdownIt 的块级规则，捕获 IndexError 并当作「未匹配」处理
（返回 False），不影响正常 markdown 的解析结果，只是让异常输入不再崩溃。

必须在 Textual 的 Markdown 组件首次创建解析器之前调用 _patch_markdown_it()
（放在 mincli.tui.app 模块导入阶段即可，Textual 每次 append 都会新建解析器）。
"""

from __future__ import annotations


def _patch_markdown_it() -> None:
    """给 markdown-it-py 的块级规则加越界保护（幂等，可重复调用）。"""
    import markdown_it.main as _mim

    if getattr(_mim, "_mincli_md_safe", False):
        return
    _mim._mincli_md_safe = True

    # 越界风险集中在块级规则（html_block 在空行处读 src[pos] 越界、
    # table/blockquote 的终止符扫描会触发它）；其余规则一并包上兜底。
    _SAFE_RULES = (
        "html_block",
        "table",
        "blockquote",
        "fence",
        "list_block",
        "hr",
        "heading",
        "lheading",
        "paragraph",
        "reference",
        "code",
    )

    _orig_init = _mim.MarkdownIt.__init__

    def _make_safe(orig):
        def _safe(state, start_line, end_line, silent):
            try:
                return orig(state, start_line, end_line, silent)
            except IndexError:
                # 越界视为规则不匹配，由后续规则继续处理
                return False

        return _safe

    def _patch_ruler(ruler) -> None:
        if ruler is None:
            return
        for rule in list(getattr(ruler, "__rules__", []) or []):
            name = getattr(rule, "name", None)
            if name in _SAFE_RULES:
                try:
                    ruler.at(name, _make_safe(rule.fn))
                except Exception:
                    pass

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        block = getattr(self, "block", None)
        if block is not None:
            _patch_ruler(getattr(block, "ruler", None))

    _mim.MarkdownIt.__init__ = _patched_init
