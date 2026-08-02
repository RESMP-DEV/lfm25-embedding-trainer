from __future__ import annotations

import ast
import copy
import inspect
import io
import shutil
import tokenize
from pathlib import Path
from types import MethodType
from typing import Any, Literal, cast

import numpy as np

PromptName = Literal["query", "document"]


def _patch_shortconv_seq_idx(model: Any) -> bool:
    """Adapt Liquid's pinned non-causal short-conv patch to Transformers 5.x."""
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ShortConv

    shortconv_modules = [module for module in model.modules() if isinstance(module, Lfm2ShortConv)]
    if not shortconv_modules:
        raise ValueError("embedding checkpoint does not contain an Lfm2ShortConv module")
    patched = False
    for module in shortconv_modules:
        slow_forward = module.slow_forward
        if "seq_idx" in inspect.signature(slow_forward).parameters:
            continue

        def compatible_slow_forward(
            self, *args, seq_idx=None, _slow_forward=slow_forward, **kwargs
        ):
            del self, seq_idx
            return _slow_forward(*args, **kwargs)

        module.slow_forward = MethodType(compatible_slow_forward, module)
        patched = True
    return patched


def _source_offset(code: str, position: tuple[int, int]) -> int:
    lines = code.splitlines(keepends=True)
    return sum(len(line) for line in lines[: position[0] - 1]) + position[1]


def _signature_span(code: str, function: ast.FunctionDef) -> tuple[int, int]:
    tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    function_index = next(
        (
            index
            for index, token in enumerate(tokens)
            if token.type == tokenize.NAME
            and token.string == "def"
            and token.start == (function.lineno, function.col_offset)
        ),
        None,
    )
    if function_index is None:
        raise ValueError("cannot locate the remote-code function signature")
    depth = 0
    opened = False
    for token in tokens[function_index:]:
        if token.type != tokenize.OP:
            continue
        if token.string in "([{":
            depth += 1
            opened = True
        elif token.string in ")]}":
            depth -= 1
        elif token.string == ":" and opened and depth == 0:
            return (
                _source_offset(code, tokens[function_index].start),
                _source_offset(code, token.end),
            )
    raise ValueError("cannot locate the end of the remote-code function signature")


def _copy_compatible_remote_code(source: Path, destination: Path) -> None:
    code = source.read_text(encoding="utf-8")
    try:
        tree = ast.parse(code, filename=str(source))
    except SyntaxError as exc:
        raise ValueError(f"cannot parse pinned LFM2 remote code at {source}") from exc
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_noncausal_shortconv_forward"
    ]
    if len(functions) != 1:
        raise ValueError(
            "pinned LFM2 remote code must define exactly one _noncausal_shortconv_forward function"
        )
    function = functions[0]
    parameters = {
        argument.arg
        for argument in (function.args.posonlyargs + function.args.args + function.args.kwonlyargs)
    }
    if "seq_idx" not in parameters:
        patched_args = copy.deepcopy(function.args)
        patched_args.kwonlyargs.append(ast.arg(arg="seq_idx"))
        patched_args.kw_defaults.append(ast.Constant(value=None))
        signature_node = ast.FunctionDef(
            name=function.name,
            args=patched_args,
            body=[ast.Pass()],
            decorator_list=[],
            returns=copy.deepcopy(function.returns),
            type_comment=None,
        )
        signature = ast.unparse(ast.fix_missing_locations(signature_node)).splitlines()[0]
        start, end = _signature_span(code, function)
        code = code[:start] + signature + code[end:]
        try:
            ast.parse(code, filename=str(destination))
        except SyntaxError as exc:
            raise ValueError(
                "cannot add Transformers 5.x seq_idx compatibility; "
                "review and pin the LFM2 remote-code revision"
            ) from exc
    destination.write_text(code, encoding="utf-8")


def resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_type(device: str) -> str:
    """Return the accelerator type for an optionally indexed device string."""
    return device.partition(":")[0]


def accelerator_backend(torch_module: Any, device: str) -> str:
    """Return the physical accelerator while retaining PyTorch's device spelling.

    PyTorch intentionally exposes ROCm devices through the ``cuda`` API. The
    distinction is still useful for receipts and backend-specific safe defaults.
    """
    if device_type(device) == "cuda" and getattr(torch_module.version, "hip", None):
        return "rocm"
    return device_type(device)


class EmbeddingEncoder:
    """Prompt-aware LFM2.5 embedding checkpoint with its native CLS pooling."""

    def __init__(self, model_id: str, revision: str = "main", device: str = "auto") -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.torch = torch
        self.device = resolve_device(device)
        self.device_type = device_type(self.device)
        self.accelerator = accelerator_backend(torch, self.device)
        self.model = SentenceTransformer(
            model_id,
            revision=revision,
            trust_remote_code=True,
            device=self.device,
        )
        _patch_shortconv_seq_idx(self.model)
        maximum_length = self.model.max_seq_length
        if maximum_length is None:
            raise ValueError("embedding checkpoint does not declare a maximum sequence length")
        self.maximum_length = maximum_length
        required_prompts = {"query", "document"}
        missing_prompts = required_prompts.difference(self.model.prompts)
        if missing_prompts:
            missing = ", ".join(sorted(missing_prompts))
            raise ValueError(
                f"model is not a prompt-aware embedding checkpoint; missing prompts: {missing}"
            )

    def encode_torch(
        self,
        texts: list[str],
        max_length: int = 512,
        prompt_name: PromptName = "document",
    ):
        if max_length > self.maximum_length:
            raise ValueError(
                f"max_length {max_length} exceeds the checkpoint limit {self.maximum_length}"
            )
        self.model.max_seq_length = max_length
        features = self.model.preprocess(
            cast(list[Any], texts), prompt=self.model.prompts[prompt_name]
        )
        features = {
            key: value.to(self.device) if isinstance(value, self.torch.Tensor) else value
            for key, value in features.items()
        }
        embeddings = self.model(features)["sentence_embedding"]
        return self.torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def encode(
        self,
        texts: list[str],
        max_length: int = 512,
        prompt_name: PromptName = "document",
    ) -> np.ndarray:
        self.model.eval()
        with self.torch.inference_mode():
            return (
                self.encode_torch(texts, max_length=max_length, prompt_name=prompt_name)
                .float()
                .cpu()
                .numpy()
            )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        # Training may deliberately truncate batches below the architecture limit.
        # Do not persist that transient value as the checkpoint's maximum length.
        self.model.max_seq_length = self.maximum_length
        self.model.save_pretrained(str(directory), safe_serialization=True)
        remote_code = Path(inspect.getfile(self.model[0].auto_model.__class__))
        destination = directory / "modeling_lfm2_bidirectional.py"
        _copy_compatible_remote_code(remote_code, destination)
        shutil.copystat(remote_code, destination)
