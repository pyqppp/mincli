TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_conversation_tree",
            "description": "查询对话节点树的结构，返回文本。不传参数时返回所有子树索引；传root只查某子树（如root='a'）；传search按标题或内容关键词搜索节点",
            "parameters": {
                "type": "object",
                "properties": {
                    "root": {
                        "type": "string",
                        "description": "可选。子对话树的字母前缀，如 'a'、'b'。只返回该子树下所有节点的详细列表。不传则返回全局索引",
                    },
                    "search": {
                        "type": "string",
                        "description": "可选。关键词，搜索节点标题和用户消息中包含该词的节点",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_conversation_nodes",
            "description": "读取指定对话节点的完整内容（用户输入+思考过程+AI回答）。可一次读多个节点，用逗号分隔节点ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_ids": {
                        "type": "string",
                        "description": "节点ID，多个用逗号分隔，如 'a1,a2,b1'",
                    }
                },
                "required": ["node_ids"],
                "additionalProperties": False,
            },
        },
    },
]
