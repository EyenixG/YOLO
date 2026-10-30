.. _ONNX:

Compile to ONNX
===============

Standalone export
-----------------

From the repository root, install the optional dependencies and export::

    python -m pip install onnx onnxruntime
    python export_onnx.py --verify

The default input is ``weights/v9-s.pt``; output is ``weights/v9-s_pre_reshape.onnx``.
Export runs on CPU with float32 input of shape ``[1, 3, 640, 640]``.
Existing output files are protected; choose a new ``--output`` to export again.

For trained repository checkpoints::

    python export_onnx.py --weights path/to/best.ckpt --model v9-s --verify

Class count is inferred from detection weights. The model name must match the
training architecture. Both plain state dictionaries and Lightning
``state_dict`` checkpoints are supported. If PyTorch refuses to deserialize
training metadata, use ``--trusted-checkpoint`` only for your own trusted file.
Weights must match the architecture strictly; mismatches stop export.

Use ``--imgsz 640 640`` for height/width, ``--batch-size 1`` for batch size,
and ``--dynamic-batch`` for variable batch size. Spatial dimensions are fixed.
Verification compares one seeded random input using rtol=0.001 and atol=0.01;
it is a numerical smoke check, not a detection accuracy evaluation.

Export before head reshape
--------------------------

All exports cut the detection heads before Anchor2Vec (reshape, softmax and DFL),
without an extra flag::

    python export_onnx.py --verify

Default output: ``weights/v9-s_pre_reshape.onnx``. Outputs are
``class_8, box_8, class_16, box_16, class_32, box_32``, all NCHW.
Class channels equal the checkpoint class count. Box channels are
``4 * reg_max`` (64 for v9-s), ordered as four consecutive distributions
of ``reg_max`` bins. At 640x640, the feature maps are 80x80, 40x40 and 20x20.
This mode needs separate DFL and box decoding downstream; the existing
FastModelLoader expects the original nine-output format and cannot directly
consume this six-output variant. Training model code is unchanged.

Automatic simplification
------------------------

Install ``onnxsim`` alongside ``onnx`` and ``onnxruntime``. Every export now
runs ONNX simplification automatically, validates equivalence with the original
ONNX using one random input, and saves the simplified graph at the chosen output
path. ``--verify`` additionally compares the final graph with PyTorch. Use
``--no-simplify`` to skip simplification. For example::

    python export_onnx.py --verify --output weights/v9-s_pre_reshape_sim.onnx
