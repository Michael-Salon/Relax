# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SGLang ServerArgs field discovery across dataclass and msgspec releases."""

import dataclasses

from relax.backends.sglang import sglang_engine


def test_server_arg_field_names_support_dataclass_and_struct(monkeypatch):
    @dataclasses.dataclass
    class LegacyArgs:
        model_path: str
        tp_size: int

    class StructArgs:
        __struct_fields__ = ("model_path", "tp_size")

    for args_type in (LegacyArgs, StructArgs):
        monkeypatch.setattr(sglang_engine, "ServerArgs", args_type)
        assert sglang_engine._server_arg_field_names() == ("model_path", "tp_size")
