# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

from scripts.tools import gemma4_chat_template_hook as hook


def _gemma4_template():
    return (
        "<|turn><channel|>{{ strip_thinking('x') }}\n"
        "    {%- if thinking_text and thinking_gate -%}\n"
        "        {{- '<|channel>thought\\n' + thinking_text + '\\n<channel|>' -}}\n"
        "    {%- endif -%}\n"
        "{%- macro format_tool_response_block(tool_name, response) -%}\n"
        "{{- '<|tool_response>' + response + '<tool_response|>' -}}\n"
        "{%- endmacro -%}\n"
        "            {%- set ns_tr_out = namespace(flag=false) -%}\n"
        "                    {{- format_tool_response_block('lookup', 'ok') -}}\n"
        "        {%- if ns.prev_message_type == 'tool_call' and not ns_tr_out.flag -%}\n"
        "            {{- '<|tool_response>' -}}\n"
    )


def test_patches_exported_gemma4_chat_template_once(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMMA4_SFT_THINKING", "1")
    monkeypatch.setenv("GEMMA4_SFT_TOOL_TURN_BOUNDARY", "1")
    path = tmp_path / "chat_template.jinja"
    path.write_text(_gemma4_template())
    args = SimpleNamespace()

    hook.patch_gemma4_chat_template(args, str(tmp_path), 1, dtype="bf16", is_lora=False)
    first = path.read_text()
    hook.patch_gemma4_chat_template(args, str(tmp_path), 2, dtype="bf16", is_lora=False)

    assert path.read_text() == first
    assert first.count("relax_gemma4_sft_thinking_v1") == 1
    assert first.count("relax_gemma4_sft_tool_turn_boundary_v1") == 1
    assert "{{- '<turn|>\\n<|turn>model\\n' -}}" in first
    assert "{{- '<|tool_response>' -}}" not in first


def test_does_not_patch_chat_template_when_gemma4_sft_mode_is_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMMA4_SFT_THINKING", raising=False)
    monkeypatch.delenv("GEMMA4_SFT_TOOL_TURN_BOUNDARY", raising=False)
    path = tmp_path / "chat_template.jinja"
    original = _gemma4_template()
    path.write_text(original)

    hook.patch_gemma4_chat_template(SimpleNamespace(), str(tmp_path), 1, dtype="bf16", is_lora=False)

    assert path.read_text() == original
