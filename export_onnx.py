"""Export this repository's YOLOv9 weights on CPU (before head reshape, without NMS).

Examples:
    python export_onnx.py
    python export_onnx.py --weights runs/train/example/last.ckpt --model v9-s
    python export_onnx.py --weights weights/v9-s.pt --verify

Dependencies: pip install onnx onnxruntime onnxsim
Only load checkpoints from trusted sources. Lightning checkpoints may require
--trusted-checkpoint to deserialize their training metadata.
"""
import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

from yolo.model.yolo import create_model
from yolo.model.module import MultiheadDetection

ROOT = Path(__file__).resolve().parent


class PreReshapeHead(nn.Module):
    """Keep head convolutions only; leave DFL reshape/softmax/expectation outside ONNX."""

    def __init__(self, head):
        super().__init__()
        self.class_conv = head.class_conv
        self.anchor_conv = head.anchor_conv

    def forward(self, features):
        return self.class_conv(features), self.anchor_conv(features)


class ExportModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        outputs = self.model(images, shortcut="Main")["Main"]
        return tuple(tensor for head in outputs for tensor in head)


def load_model(weights, model_name, trusted=False):
    checkpoint = torch.load(weights, map_location="cpu", weights_only=not trusted)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a state_dict or Lightning checkpoint from this repository")
    state = checkpoint.get("state_dict", checkpoint)
    if any(key.startswith("model.model.") for key in state):
        state = {key.removeprefix("model.model."): value for key, value in state.items()
                 if key.startswith("model.model.")}
    elif any(key.startswith("model.") for key in state):
        state = {key.removeprefix("model."): value for key, value in state.items()
                 if key.startswith("model.")}
    counts = {value.shape[0] for key, value in state.items()
              if key.endswith("class_conv.2.weight")}
    if len(counts) != 1:
        raise ValueError("Cannot infer class count; expected YOLOv9 detection weights")
    classes = counts.pop()
    cfg = OmegaConf.load(ROOT / "yolo" / "config" / "model" / f"{model_name}.yaml")
    model = create_model(cfg, weight_path=False, class_num=classes)
    # Strict loading prevents silently exporting randomly initialized layers.
    model.model.load_state_dict(state, strict=True)
    for layer in model.model:
        if isinstance(layer, MultiheadDetection):
            layer.heads = nn.ModuleList([PreReshapeHead(head) for head in layer.heads])
    return ExportModel(model).cpu().float().eval(), classes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/v9-s.pt")
    parser.add_argument("--model", default="v9-s", help="Model YAML name; must match training architecture")
    parser.add_argument("--output", type=Path, help="Default: <weights stem>_pre_reshape.onnx")
    parser.add_argument("--imgsz", type=int, nargs=2, default=[640, 640], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Compare ONNX Runtime and PyTorch outputs")
    parser.add_argument("--trusted-checkpoint", action="store_true")
    parser.add_argument("--no-simplify", action="store_true", help="Skip automatic ONNX simplification")
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Weights do not exist: {args.weights}")
    if args.batch_size < 1 or any(size <= 0 or size % 32 for size in args.imgsz):
        parser.error("Batch size must be positive; image dimensions must be positive multiples of 32")
    try:
        import onnx
    except ImportError:
        parser.error("Install ONNX: python -m pip install onnx (and onnxruntime for --verify)")
    if not args.no_simplify:
        try:
            from onnxsim import simplify
        except ImportError:
            parser.error("Install simplifier: python -m pip install onnxsim onnxruntime")
    if args.verify:
        import onnxruntime as ort
        import numpy as np
    suffix = "_pre_reshape.onnx"
    output = args.output or args.weights.with_name(args.weights.stem + suffix)
    if output.resolve() == args.weights.resolve():
        parser.error("Output must differ from input weights")
    if output.exists():
        parser.error(f"Output already exists; choose another --output: {output}")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(0)
    model, classes = load_model(args.weights, args.model, args.trusted_checkpoint)
    images = torch.rand(args.batch_size, 3, *args.imgsz)
    kinds = ("class", "box")
    names = [f"{kind}_{scale}" for scale in (8, 16, 32) for kind in kinds]
    dynamic = {name: {0: "batch"} for name in ["input", *names]} if args.dynamic_batch else None
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        expected = model(images)
        torch.onnx.export(model, images, str(output), input_names=["input"], output_names=names,
                          opset_version=args.opset, dynamic_axes=dynamic, dynamo=False)
    onnx.checker.check_model(str(output))
    if not args.no_simplify:
        original = onnx.load(str(output))
        simplified, valid = simplify(original, check_n=1)
        if not valid:
            raise RuntimeError(f"ONNX simplification validation failed; original export remains at {output}")
        onnx.checker.check_model(simplified)
        onnx.save(simplified, str(output))
        print(f"ONNX simplified: {len(original.graph.node)} -> {len(simplified.graph.node)} nodes")
    if args.verify:
        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        actual = session.run(None, {"input": images.numpy()})
        assert len(actual) == len(expected)
        for name, reference, result in zip(names, expected, actual):
            # Raw logits can differ slightly after ONNX convolution/BN fusion.
            np.testing.assert_allclose(result, reference.numpy(), rtol=1e-3, atol=1e-2, err_msg=name)
            print(f"{name}: max absolute error={np.max(np.abs(result - reference.numpy())):.6g}")
        print("ONNX Runtime numerical verification passed")
    print(f"Saved: {output} | classes={classes} | input={tuple(images.shape)}")
    print("Outputs: 3 scales x (class logits, box logits), all NCHW; no head reshape/DFL/NMS")


if __name__ == "__main__":
    main()
