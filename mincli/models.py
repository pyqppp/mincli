import re
import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any


@dataclass
class StreamResult:
    content: Optional[str] = None
    reasoning: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: Optional[List[Dict]] = None


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
            tool_messages=data.get("tool_messages", []),
        )


class ConversationTree:

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.nodes: Dict[str, ConversationNode] = {}
        self.root: Optional[ConversationNode] = None
        self.current_node: Optional[ConversationNode] = None
        self.subtree_titles: Dict[str, str] = {}

    def _generate_child_id(self, parent: ConversationNode) -> str:
        used_ids = set(self.nodes.keys())

        if not parent.children:
            match = re.match(r'^([a-z]+)(\d+)$', parent.id)
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

        used_letters = set()
        for nid in used_ids:
            match = re.match(r'^([a-z]+)\d+$', nid)
            if match:
                used_letters.add(match.group(1))

        for letter in range(ord('a'), ord('z') + 1):
            l = chr(letter)
            if l not in used_letters:
                candidate = f"{l}1"
                if candidate not in used_ids:
                    return candidate
                num = 2
                while f"{l}{num}" in used_ids:
                    num += 1
                return f"{l}{num}"

        return f"z_{datetime.datetime.now().strftime('%H%M%S')}"

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
        msgs = node.get_messages(self)
        return [{"role": "system", "content": self.system_prompt}] + msgs

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
        match = re.match(r'^([a-z]+)', node.id)
        return match.group(1) if match else None

    def count_subtree_nodes(self, prefix: str) -> int:
        root_id = next((nid for nid in self.nodes
                        if nid.startswith(prefix)
                        and self.nodes[nid].parent_id == "main"), None)
        if not root_id:
            return 0
        descendants = set()
        self._collect_descendants(self.nodes[root_id], descendants)
        return len(descendants)

    def get_subtree_branches(self, prefix: str) -> List[str]:
        root_id = next((nid for nid in self.nodes
                        if nid.startswith(prefix)
                        and self.nodes[nid].parent_id == "main"), None)
        if not root_id:
            return []
        descendants = set()
        self._collect_descendants(self.nodes[root_id], descendants)
        branches = set()
        for nid in descendants:
            m = re.match(r'^([a-z]+)', nid)
            if m and m.group(1) != prefix:
                branches.add(m.group(1))
        return sorted(branches)

    def render_tree(self, highlight_id: Optional[str] = None,
                    active_subtree: Optional[str] = None) -> Any:
        from rich.tree import Tree as RichTree

        if not self.root:
            return RichTree("[空树]")
        root_tree = RichTree(f"{self.root.id}: {self.root.title}")

        if active_subtree:
            for child in self.root.children:
                prefix = self._get_subtree_root_prefix(child.id)
                if prefix == active_subtree:
                    label = f"{child.id}: {child.title}"
                    if child.id == highlight_id:
                        label = f"[bold cyan]➤ {label}[/bold cyan]"
                    child_tree = root_tree.add(label)
                    self._add_node_to_rich_tree(child_tree, child, highlight_id)
                elif prefix and prefix in self.subtree_titles:
                    branches = self.get_subtree_branches(prefix)
                    label = f"{prefix}：{self.subtree_titles[prefix]}"
                    if branches:
                        label += f"（{'，'.join(branches)}）"
                    root_tree.add(label)
                else:
                    label = f"{child.id}: {child.title}"
                    child_tree = root_tree.add(label)
                    self._add_node_to_rich_tree(child_tree, child, highlight_id)
        else:
            self._add_node_to_rich_tree(root_tree, self.root, highlight_id)

        return root_tree

    def _add_node_to_rich_tree(self, rich_node, node: ConversationNode,
                               highlight_id: Optional[str]):
        for child in node.children:
            label = f"{child.id}: {child.title}"
            if child.id == highlight_id:
                label = f"[bold cyan]➤ {label}[/bold cyan]"
            child_tree = rich_node.add(label)
            self._add_node_to_rich_tree(child_tree, child, highlight_id)

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
        return tree
