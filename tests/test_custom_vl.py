"""Dependency-free tests of the actual VL methods (no models or ComfyUI needed)."""
import ast
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qwenvl_custom import (
    CUSTOM_SYSTEM_PROMPT_INPUT, remaining_custom_tokens, vl_preset_options,
)


def load_method(filename, classname, method, namespace):
    tree = ast.parse((ROOT / filename).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == classname)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    fn.decorator_list = []
    exec(compile(ast.Module(body=[fn], type_ignores=[]), filename, "exec"), namespace)
    return namespace[method]


class CustomVLTests(unittest.TestCase):
    def namespace(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = False
        torch.is_tensor.return_value = False
        return dict(
            DEFAULT_DURATION="6s", torch=torch, PROMPT_CACHE={}, LAST_SAVED_PROMPT="",
            SYSTEM_PROMPTS={"IMG › Detailed": "existing template"},
            expand_wildcard_tokens=lambda p: p.replace("__tag__", "expanded"),
            resolve_vl_preset=lambda p, d: (p, p),
            get_image_hash=lambda i: None, get_video_hash=lambda i: None,
            get_cache_key=MagicMock(return_value="test-key"),
            add_danbooru_guidance=lambda p, n: p,
            CAMERA_TAG_DESCRIPTIONS={"PAN": "pan"},
            camera_directive_location=lambda *a: "camera instruction",
            normalize_minimax_output=MagicMock(return_value="normalized"),
            save_prompt_cache=lambda: None,
            ensure_cuda_vram_headroom=lambda *a, **kw: None,
            remaining_custom_tokens=remaining_custom_tokens,
            Image=types.SimpleNamespace(Image=type("PILImage", (), {})),
        )

    def test_options_preserve_presets_and_always_include_custom_once(self):
        self.assertEqual(vl_preset_options(["one", "two"]), ["one", "two", "custom"])
        self.assertEqual(vl_preset_options(["custom"]), ["custom"])
        self.assertIn("custom", vl_preset_options([]))

    def test_budget(self):
        self.assertEqual(remaining_custom_tokens(31000, 8192), 1000)
        self.assertEqual(remaining_custom_tokens(10, 100), 100)
        for count in (32000, 32001):
            with self.assertRaisesRegex(ValueError, "leave room"):
                remaining_custom_tokens(count, 8192)

    def test_custom_run_bypasses_preset_camera_wildcards_and_normalization(self):
        for filename, classname in (("AILab_QwenVL.py", "QwenVLBase"),
                                    ("AILab_QwenVL_GGUF.py", "QwenVLGGUFBase")):
            with self.subTest(backend=classname):
                ns = self.namespace()
                run = load_method(filename, classname, "run", ns)
                node = MagicMock()
                node.is_qwen35 = True
                node.generate.return_value = "raw answer"
                node._invoke.return_value = "raw answer"
                args = dict(model_name="test", preset_prompt="custom", prompt="__tag__ [PAN]",
                            image=None, image2=None, image3=None, video=None, frame_count=16,
                            max_tokens=32000, temperature=0.6, top_p=0.9,
                            repetition_penalty=1.0, seed=1, keep_model_loaded=True,
                            device="auto", camera_tag="PAN", custom_system_prompt="my rules")
                if classname == "QwenVLBase":
                    args.update(quantization="none", num_beams=1,
                                attention_mode="auto", use_torch_compile=False)
                else:
                    args["ctx"] = 65536
                self.assertEqual(run(node, **args), ("raw answer",))
                ns["normalize_minimax_output"].assert_not_called()
                if classname == "QwenVLBase":
                    call = node.generate.call_args
                    self.assertEqual(call.args[0], "__tag__ [PAN]")
                    self.assertEqual(call.kwargs["custom_system_prompt"], "my rules")
                else:
                    call = node._invoke.call_args.kwargs
                    self.assertEqual(call["user_prompt"], "__tag__ [PAN]")
                    self.assertEqual(call["system_prompt"], "my rules")
                    self.assertTrue(call["raw_output"])
                    self.assertEqual(node._load_model.call_args.kwargs["ctx"], 32000)

    def test_builtin_presets_keep_existing_behavior(self):
        for filename, classname in (("AILab_QwenVL.py", "QwenVLBase"),
                                    ("AILab_QwenVL_GGUF.py", "QwenVLGGUFBase")):
            with self.subTest(backend=classname):
                ns = self.namespace()
                run = load_method(filename, classname, "run", ns)
                node = MagicMock()
                node.is_qwen35 = False
                node.generate.return_value = "answer"
                node._invoke.return_value = "answer"
                args = dict(model_name="test", preset_prompt="IMG › Detailed", prompt="__tag__",
                            image=None, image2=None, image3=None, video=None, frame_count=16,
                            max_tokens=8192, temperature=0.6, top_p=0.9, repetition_penalty=1.0,
                            seed=1, keep_model_loaded=True, device="auto", camera_tag="PAN",
                            custom_system_prompt="must be ignored")
                if classname == "QwenVLBase":
                    args.update(quantization="none", num_beams=1,
                                attention_mode="auto", use_torch_compile=False)
                self.assertEqual(run(node, **args), ("normalized",))
                ns["normalize_minimax_output"].assert_called_once()
                if classname == "QwenVLBase":
                    call = node.generate.call_args
                    prompt = call.args[0]
                    self.assertIsNone(call.kwargs["custom_system_prompt"])
                else:
                    call = node._invoke.call_args.kwargs
                    prompt = call["user_prompt"]
                    self.assertNotEqual(call["system_prompt"], "must be ignored")
                self.assertIn("expanded", prompt)
                self.assertIn("existing template", prompt)
                self.assertIn("FINAL CAMERA DIRECTIVE", prompt)

    def test_hf_actual_chat_roles_and_encoded_budget(self):
        ns = self.namespace()
        generate = load_method("AILab_QwenVL.py", "QwenVLBase", "generate", ns)
        node = MagicMock()
        node.is_qwen35 = True
        node.model.config = types.SimpleNamespace(max_position_embeddings=65536)
        node.model.parameters.return_value = iter([types.SimpleNamespace(device="cpu")])
        node.processor.return_value = {"input_ids": types.SimpleNamespace(shape=(1, 31000))}
        node.tokenizer.decode.return_value = "answer"
        result = generate(node, "user text", None, None, None, 16, 8192, 0.6, 0.9, 1, 1.0,
                          custom_system_prompt="system rules")
        self.assertEqual(result, "answer")
        messages = node.processor.apply_chat_template.call_args.args[0]
        self.assertEqual(messages[0], {"role": "system", "content": "system rules"})
        self.assertEqual(messages[1]["content"], [{"type": "text", "text": "user text"}])
        self.assertEqual(node.model.generate.call_args.kwargs["max_new_tokens"], 1000)

    def test_all_vl_schemas_and_wrappers_accept_custom_prompt(self):
        # Also check that the new optional widget is last, preserving saved values.
        for filename, classes in (
            ("AILab_QwenVL.py", ("AILab_QwenVL", "AILab_QwenVL_Advanced")),
            ("AILab_QwenVL_GGUF.py", ("AILab_QwenVL_GGUF", "AILab_QwenVL_GGUF_Advanced")),
            ("QwenVL_Unified.py", ("QwenVL_Unified", "QwenVL_Unified_Advanced")),
        ):
            for classname in classes:
                with self.subTest(node=classname):
                    ns = self.namespace()
                    ns.update(
                        PRESET_PROMPTS=["IMG › Detailed"], HF_VL_MODELS={"test": {}},
                        GGUF_VL_CATALOG={"models": {}},
                        Quantization=types.SimpleNamespace(get_values=lambda: ["fp16"],
                                                          FP16=types.SimpleNamespace(value="fp16")),
                        ATTENTION_MODES=["auto"], CAMERA_TAG_OPTIONS=["None"], CAMERA_TAG_TOOLTIP="",
                        TOOLTIPS=types.SimpleNamespace(),
                        DURATION_OPTIONS=["6s"], CUSTOM_SYSTEM_PROMPT_INPUT=CUSTOM_SYSTEM_PROMPT_INPUT,
                        vl_preset_options=vl_preset_options,
                        _combined_model_list=lambda: ["test"], _default_model=lambda: "test",
                    )
                    ns["TOOLTIPS"] = type("Tooltips", (dict,), {"__missing__": lambda self, k: ""})()
                    schema = load_method(filename, classname, "INPUT_TYPES", ns)(object)
                    self.assertIn("custom", schema["required"]["preset_prompt"][0])
                    self.assertEqual(list(schema["optional"])[-1], "custom_system_prompt")
                    self.assertEqual(schema["required"]["max_tokens"][1]["max"], 32000)
                    process = load_method(filename, classname, "process", ns)
                    import inspect
                    signature = inspect.signature(process)
                    self.assertEqual(signature.parameters["custom_system_prompt"].default, "")
                    args = {name: None for name, p in signature.parameters.items()
                            if p.default is inspect.Parameter.empty and name != "self"}
                    args.update(model_name="HF: test", custom_system_prompt="rules")
                    ns.update(GGUF_PREFIX="GGUF: ", HF_PREFIX="HF: ")
                    node = MagicMock()
                    process(node, **args)
                    called = node._hf.process if filename == "QwenVL_Unified.py" else node.run
                    self.assertEqual(called.call_args.kwargs["custom_system_prompt"], "rules")
                    if filename == "QwenVL_Unified.py":
                        args["model_name"] = "GGUF: test"
                        process(node, **args)
                        self.assertEqual(node._gguf.process.call_args.kwargs["custom_system_prompt"], "rules")


if __name__ == "__main__":
    unittest.main()
