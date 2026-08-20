import re
import itertools
import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Iterator


@dataclass
class StreamResult:
    content: Optional[str] = None
    reasoning: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    # DeepSeek usage：上下文缓存命中/未命中的输入 token 数
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    tool_calls: Optional[List[Dict]] = None
    error: str = ""


@dataclass
class ConversationNode:
    id: str
    parent_id: Optional[str] = None
    user_msg: str = ""
    assistant_msg: str = ""
    reasoning: str = ""
    title: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    # 本节点各次 API 请求的上下文缓存统计（DeepSeek usage 字段）
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    children: List['ConversationNode'] = field(default_factory=list)
    cached_messages: Optional[List[Dict]] = None
    tool_messages: List[Dict] = field(default_factory=list)

    def get_messages(self, tree: 'ConversationTree') -> List[Dict]:
        if self.cached_messages is not None:
            return self.cached_messages

        msgs = []
        if self.parent_id:
            parent = tree.nodes.get(self.parent_id)
            if parent:
                msgs = parent.get_messages(tree).copy()

        msgs.append({"role": "user", "content": self.user_msg})
        for tm in self.tool_messages:
            msgs.append(tm)
        if self.assistant_msg:
            assistant_msg = {"role": "assistant", "content": self.assistant_msg}
            if self.reasoning:
                assistant_msg["reasoning_content"] = self.reasoning
            msgs.append(assistant_msg)

        self.cached_messages = msgs
        return msgs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "user_msg": self.user_msg,
            "assistant_msg": self.assistant_msg,
            "reasoning": self.reasoning,
            "title": self.title,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
            "tool_messages": self.tool_messages,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationNode':
        return cls(
            id=data["id"],
            parent_id=data["parent_id"],
            user_msg=data["user_msg"],
            assistant_msg=data["assistant_msg"],
            reasoning=data.get("reasoning", ""),
            title=data["title"],
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            cache_hit_tokens=data.get("cache_hit_tokens", 0),
            cache_miss_tokens=data.get("cache_miss_tokens", 0),
            tool_messages=data.get("tool_messages", []),
        )


_ID_ALPHABET_LOWER = "abcdefghijklmnopqrstuvwxyz"
_ID_ALPHABET_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# 压缩摘要注入消息时的固定前缀（标记该消息为历史摘要，便于模型理解）
COMPACTION_PREFIX = "【以下是本对话早期内容的详细摘要，原文已压缩（如需某处细节可追问展开）】\n\n"


def _iter_id_prefixes() -> Iterator[str]:
    """节点字母前缀序列：a-z → A-Z → aa-az, ba-bz … za-zz → AA-ZZ → aaa …"""
    length = 1
    while True:
        for alphabet in (_ID_ALPHABET_LOWER, _ID_ALPHABET_UPPER):
            for combo in itertools.product(alphabet, repeat=length):
                yield "".join(combo)
        length += 1


class ConversationTree:

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.nodes: Dict[str, ConversationNode] = {}
        self.root: Optional[ConversationNode] = None
        self.current_node: Optional[ConversationNode] = None
        self.subtree_titles: Dict[str, str] = {}
        # 上下文压缩：{"summary": str, "boundary_id": str, "next_input_tokens": int}。
        # boundary_id 为最后一个被压缩进摘要的节点；其后（含其后分支）仍按原文发送；
        # next_input_tokens 为压缩报告 after_tokens（状态条「下次输入」压缩后采用）。
        self.compaction: Optional[Dict[str, Any]] = None

    def _generate_child_id(self, parent: ConversationNode) -> str:
        used_ids = set(self.nodes.keys())

        if not parent.children:
            match = re.match(r'^([A-Za-z]+)(\d+)$', parent.id)
            if match:
                prefix = match.group(1)
                num = int(match.group(2)) + 1
                candidate = f"{prefix}{num}"
                while candidate in used_ids:
                    num += 1
                    candidate = f"{prefix}{num}"
                return candidate
            else:
                candidate = "a1"
                while candidate in used_ids:
                    num = int(re.search(r'\d+$', candidate).group()) + 1
                    candidate = f"a{num}"
                return candidate

        used_prefixes = set()
        for nid in used_ids:
            match = re.match(r'^([A-Za-z]+)\d+$', nid)
            if match:
                used_prefixes.add(match.group(1))

        for prefix in _iter_id_prefixes():
            if prefix not in used_prefixes:
                num = 1
                while f"{prefix}{num}" in used_ids:
                    num += 1
                return f"{prefix}{num}"

        return f"z_{datetime.datetime.now().strftime('%H%M%S')}"

    def _node_letter_prefix(self, node_id: str) -> Optional[str]:
        """返回节点 ID 的字母前缀（如 'a1'→'a'、'ab1'→'ab'），非分支节点返回 None。"""
        match = re.match(r'^([A-Za-z]+)\d+$', node_id)
        return match.group(1) if match else None

    def create_root(self, user_msg: str, assistant_msg: str, reasoning: str,
                    title: str, input_tokens: int, output_tokens: int) -> ConversationNode:
        node = ConversationNode(
            id="main",
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            reasoning=reasoning,
            title=title,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.nodes[node.id] = node
        self.root = node
        self.current_node = node
        return node

    def add_child(self, parent: ConversationNode, user_msg: str, assistant_msg: str,
                  reasoning: str, title: str, input_tokens: int, output_tokens: int) -> ConversationNode:
        child_id = self._generate_child_id(parent)
        node = ConversationNode(
            id=child_id,
            parent_id=parent.id,
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            reasoning=reasoning,
            title=title,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.nodes[node.id] = node
        parent.children.append(node)
        return node

    def get_messages_for_node(self, node: ConversationNode) -> List[Dict]:
        """构建发送给 LLM 的消息（含 system）。

        若存在压缩摘要且 node 位于压缩边界（boundary）上或其后，
        则用摘要替换 boundary 及之前全部节点的原始消息，仅保留 boundary
        之后的原始消息，从而缩短上下文。
        """
        msgs = node.get_messages(self)
        comp = self.compaction
        if comp and comp.get("boundary_id"):
            boundary = self.nodes.get(comp["boundary_id"])
            if boundary is not None and self._is_at_or_after(node, boundary.id):
                raw_prefix = boundary.get_messages(self)
                summary_block = [
                    {"role": "user", "content": COMPACTION_PREFIX + comp["summary"]}
                ]
                msgs = summary_block + msgs[len(raw_prefix):]
        return [{"role": "system", "content": self.system_prompt}] + msgs

    def _is_at_or_after(self, node: ConversationNode, ancestor_id: str) -> bool:
        """node 是否位于 ancestor_id 所在节点上或其下游（含该节点自身）。"""
        cur: Optional[ConversationNode] = node
        while cur is not None:
            if cur.id == ancestor_id:
                return True
            cur = self.nodes.get(cur.parent_id) if cur.parent_id else None
        return False

    def clear_compaction(self) -> bool:
        """清除压缩摘要，恢复发送完整原始消息。返回是否真的清除了。"""
        if self.compaction:
            self.compaction = None
            return True
        return False

    def switch_to_node(self, node_id: str) -> bool:
        if node_id in self.nodes:
            self.current_node = self.nodes[node_id]
            return True
        return False

    def delete_node(self, node_id: str) -> bool:
        if node_id not in self.nodes:
            return False
        node = self.nodes[node_id]
        to_delete = set()
        self._collect_descendants(node, to_delete)

        if node.parent_id:
            parent = self.nodes.get(node.parent_id)
            if parent:
                parent.children = [c for c in parent.children if c.id != node_id]

        for nid in to_delete:
            del self.nodes[nid]

        if self.root and self.root.id in to_delete:
            self.root = None

        if self.current_node and self.current_node.id in to_delete:
            if node.parent_id and node.parent_id in self.nodes:
                self.current_node = self.nodes[node.parent_id]
            else:
                self.current_node = self.root

        return True

    def _collect_descendants(self, node: ConversationNode, result: set):
        result.add(node.id)
        for child in node.children:
            self._collect_descendants(child, result)

    def _get_subtree_root_prefix(self, node_id: str) -> Optional[str]:
        node = self.nodes.get(node_id)
        if not node or not self.root or node.id == "main":
            return None
        while node.parent_id != "main":
            parent = self.nodes.get(node.parent_id)
            if not parent:
                return None
            node = parent
        match = re.match(r'^([A-Za-z]+)', node.id)
        return match.group(1) if match else None

    def count_subtree_nodes(self, prefix: str) -> int:
        root_id = next((nid for nid in self.nodes
                        if self._node_letter_prefix(nid) == prefix
                        and self.nodes[nid].parent_id == "main"), None)
        if not root_id:
            return 0
        descendants = set()
        self._collect_descendants(self.nodes[root_id], descendants)
        return len(descendants)

    def get_subtree_branches(self, prefix: str) -> List[str]:
        root_id = next((nid for nid in self.nodes
                        if self._node_letter_prefix(nid) == prefix
                        and self.nodes[nid].parent_id == "main"), None)
        if not root_id:
            return []
        descendants = set()
        self._collect_descendants(self.nodes[root_id], descendants)
        branches = set()
        for nid in descendants:
            p = self._node_letter_prefix(nid)
            if p and p != prefix:
                branches.add(p)
        return sorted(branches)

    def render_tree(self, highlight_id: Optional[str] = None,
                    active_subtree: Optional[str] = None) -> str:
        """渲染对话树为纯文本（无 Rich 依赖），返回多行字符串。"""
        if not self.root:
            return "[空树]"
        lines: list[str] = []
        root_label = f"main: {self.root.title}"
        if self.root.id == highlight_id:
            root_label = f"➤ {root_label}"
        lines.append(root_label)

        if active_subtree:
            for child in self.root.children:
                prefix = self._get_subtree_root_prefix(child.id)
                if prefix == active_subtree:
                    label = f"{child.id}: {child.title}"
                    if child.id == highlight_id:
                        label = f"➤ {label}"
                    lines.append("  " + label)
                    self._add_node_lines(lines, child, highlight_id, 2)
                elif prefix and prefix in self.subtree_titles:
                    branches = self.get_subtree_branches(prefix)
                    label = f"{prefix}：{self.subtree_titles[prefix]}"
                    if branches:
                        label += f"（{'，'.join(branches)}）"
                    lines.append("  " + label)
                else:
                    label = f"{child.id}: {child.title}"
                    if child.id == highlight_id:
                        label = f"➤ {label}"
                    lines.append("  " + label)
                    self._add_node_lines(lines, child, highlight_id, 2)
        else:
            self._add_node_lines(lines, self.root, highlight_id, 1)

        return "\n".join(lines)

    def _count_linear_chain(self, node: ConversationNode) -> int:
        count = 0
        while len(node.children) == 1:
            count += 1
            node = node.children[0]
        return count

    def _add_node_lines(self, lines: list, node: ConversationNode,
                        highlight_id: Optional[str], depth: int) -> None:
        for child in node.children:
            chain_len = self._count_linear_chain(child) + 1
            if chain_len >= 5:
                # 线性链折叠显示：链上各节点各自占一行（纯文本下更清晰）
                last = child
                for i in range(chain_len):
                    label = f"{last.id}: {last.title}"
                    if last.id == highlight_id:
                        label = f"➤ {label}"
                    lines.append("  " * depth + label)
                    if i < chain_len - 1:
                        last = last.children[0]
                self._add_node_lines(lines, last, highlight_id, depth + 1)
            else:
                label = f"{child.id}: {child.title}"
                if child.id == highlight_id:
                    label = f"➤ {label}"
                lines.append("  " * depth + label)
                self._add_node_lines(lines, child, highlight_id, depth + 1)



    def get_branch_total_tokens(self, node_id: str) -> Tuple[int, int]:
        total_in = 0
        total_out = 0
        node = self.nodes.get(node_id)
        while node:
            total_in += node.input_tokens
            total_out += node.output_tokens
            node = self.nodes.get(node.parent_id) if node.parent_id else None
        return total_in, total_out

    def to_dict(self) -> Dict[str, Any]:
        nodes_data = {}
        for nid, node in self.nodes.items():
            nodes_data[nid] = node.to_dict()
        return {
            "system_prompt": self.system_prompt,
            "nodes": nodes_data,
            "root_id": self.root.id if self.root else None,
            "current_node_id": self.current_node.id if self.current_node else None,
            "subtree_titles": self.subtree_titles,
            "compaction": self.compaction,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTree':
        tree = cls(system_prompt=data["system_prompt"])
        for nid, node_data in data["nodes"].items():
            node = ConversationNode.from_dict(node_data)
            tree.nodes[nid] = node
        for nid, node in tree.nodes.items():
            if node.parent_id:
                parent = tree.nodes.get(node.parent_id)
                if parent:
                    parent.children.append(node)
        root_id = data.get("root_id")
        if root_id:
            tree.root = tree.nodes.get(root_id)
        current_id = data.get("current_node_id")
        if current_id:
            tree.current_node = tree.nodes.get(current_id)
        tree.subtree_titles = data.get("subtree_titles", {})
        tree.compaction = data.get("compaction")
        return tree
