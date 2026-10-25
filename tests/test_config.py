"""配置层测试：生成、回退、写回、提示词文件。"""

import os

import pytest

from mincli import config


@pytest.fixture()
def cfg_path(tmp_path, monkeypatch):
    path = str(tmp_path / "config.toml")
    monkeypatch.setenv("MINCLI_CONFIG_PATH", path)
    monkeypatch.setenv("MINCLI_PROMPT_PATH", str(tmp_path / "prompt.md"))
    monkeypatch.setenv("MINCLI_ATTACHMENTS_PATH", str(tmp_path / "attachments"))
    return path


def test_ensure_config_generates_file_and_prompt(cfg_path, tmp_path):
    cfg = config.ensure_config(cfg_path)
    assert os.path.exists(cfg_path)
    assert os.path.exists(cfg.prompt_file)
    assert cfg.model == config.MODEL_FLASH
    assert cfg.files_mode == "off"
    assert cfg.web_enabled is True
    assert cfg.recall_enabled is False
    assert cfg.inline_max_chars == config.INLINE_MAX_CHARS_DEFAULT

    text = open(cfg_path, encoding="utf-8").read()
    assert "[files]" in text and "[web]" in text and "[prompt]" in text
    assert "# mincli 配置文件" in text


def test_ensure_config_keeps_existing_prompt(cfg_path, tmp_path):
    prompt = tmp_path / "prompt.md"
    prompt.write_text("自定义提示词", encoding="utf-8")
    cfg = config.Config(prompt_file=str(prompt))
    config.save_config(cfg, cfg_path)

    loaded = config.ensure_config(cfg_path)
    assert open(loaded.prompt_file, encoding="utf-8").read() == "自定义提示词"


def test_missing_file_gives_defaults_without_writing(tmp_path):
    path = str(tmp_path / "nope.toml")
    cfg = config.load_config(path)
    assert cfg.files_mode == "off"
    assert not os.path.exists(path)


def test_invalid_values_fall_back(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(
        """
model = "gpt-4o"
temperature = 9.5
thinking = "yes"
effort = "ultra"

[files]
mode = "write"
inline_max_chars = 3

[web]
enabled = "false"
""",
        encoding="utf-8",
    )
    cfg = config.load_config(str(path))
    assert cfg.model == config.MODEL_FLASH
    assert cfg.temperature == config.TEMPERATURE_MAX
    assert cfg.thinking is True
    assert cfg.effort == config.DEFAULT_EFFORT
    assert cfg.files_mode == "off"
    assert cfg.inline_max_chars == config.INLINE_MAX_CHARS_MIN
    assert cfg.web_enabled is False


def test_broken_toml_falls_back(tmp_path):
    path = tmp_path / "broken.toml"
    path.write_text("model = [", encoding="utf-8")
    cfg = config.load_config(str(path))
    assert cfg.model == config.MODEL_FLASH
    assert cfg.web_enabled is True


def test_write_back_round_trip(cfg_path):
    cfg = config.ensure_config(cfg_path)
    cfg = cfg.with_files_mode("direct").with_web(False).with_recall(True)
    cfg.inline_max_chars = 5000
    config.save_config(cfg, cfg_path)

    reloaded = config.load_config(cfg_path)
    assert reloaded.files_mode == "direct"
    assert reloaded.web_enabled is False
    assert reloaded.recall_enabled is True
    assert reloaded.inline_max_chars == 5000

    text = open(cfg_path, encoding="utf-8").read()
    assert '# mincli 配置文件' in text
    assert 'mode = "direct"' in text
    assert "enabled = false" in text


def test_file_mode_helpers():
    cfg = config.Config()
    assert cfg.files_read_enabled is False
    assert cfg.files_label == "关闭"

    cfg = cfg.with_files_mode("read")
    assert cfg.files_read_enabled is True
    assert cfg.files_write_enabled is False

    cfg = cfg.with_files_mode("confirm")
    assert cfg.files_write_enabled is True
    assert cfg.files_write_needs_confirm is True

    cfg = cfg.with_files_mode("direct")
    assert cfg.files_write_needs_confirm is False

    # 非法模式不改变状态
    assert cfg.with_files_mode("oops").files_mode == "direct"


def test_system_prompt_reads_file(tmp_path):
    prompt = tmp_path / "p.md"
    prompt.write_text("  你是测试助手  ", encoding="utf-8")
    cfg = config.Config(prompt_file=str(prompt))
    assert config.system_prompt(cfg) == "你是测试助手"


def test_system_prompt_falls_back_when_missing(tmp_path):
    cfg = config.Config(prompt_file=str(tmp_path / "missing.md"))
    assert config.system_prompt(cfg) == config.DEFAULT_PROMPT.strip()


def test_default_prompt_has_no_emoji_requirement():
    # 提示词不应把 emoji 当状态标记（全局约定）
    assert "[OK]" not in config.DEFAULT_PROMPT
    assert "✅" not in config.DEFAULT_PROMPT


def test_default_prompt_matches_current_tools():
    """默认提示词必须与当前工具面和命令面一致，不能提到已删除的功能。"""
    prompt = config.DEFAULT_PROMPT
    for tool in (
        "read_attachment",
        "read_file",
        "list_directory",
        "write_file",
        "edit_file",
        "web_search",
        "web_fetch",
        "query_conversation_tree",
        "read_conversation_nodes",
    ):
        assert tool in prompt, f"缺少工具说明：{tool}"

    for command in ("/file", "/web", "/recall", "/model", "/thinking", "/help"):
        assert command in prompt, f"缺少命令入口：{command}"

    for removed in ("/set", "/import", "/wf", "/mcp", "execute_command", "MCP", "工作流"):
        assert removed not in prompt, f"提示词仍提到已删除功能：{removed}"


def test_prompt_file_gets_new_default(tmp_path):
    path = tmp_path / "prompt.md"
    cfg = config.Config(prompt_file=str(path))
    config.ensure_prompt_file(cfg)
    assert path.read_text(encoding="utf-8") == config.DEFAULT_PROMPT


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    assert config.api_key() == "sk-test"
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert config.api_key() == ""


def test_webpage_max_length_clamped(monkeypatch):
    monkeypatch.delenv("MINCLI_WEBPAGE_MAX_LENGTH", raising=False)
    assert config.webpage_max_length() == config.WEBPAGE_MAX_LENGTH_DEFAULT

    monkeypatch.setenv("MINCLI_WEBPAGE_MAX_LENGTH", "1234")
    assert config.webpage_max_length() == 1234

    monkeypatch.setenv("MINCLI_WEBPAGE_MAX_LENGTH", "999999")
    assert config.webpage_max_length() == config.WEBPAGE_MAX_LENGTH_LIMIT

    monkeypatch.setenv("MINCLI_WEBPAGE_MAX_LENGTH", "abc")
    assert config.webpage_max_length() == config.WEBPAGE_MAX_LENGTH_DEFAULT
