"""Rewrite the exported graphs with phoonnx's external-data naming.

``torch.onnx.export`` writes the tensor blob as ``<name>.onnx.data``. phoonnx's model
manager downloads the sidecar as ``<name>.onnx_data``, so the graphs have to reference
that name instead. Run this after the four export scripts.
"""
import glob
import os

import onnx

from _paths import out_dir


def main() -> None:
    directory = out_dir()
    for path in sorted(glob.glob(os.path.join(directory, "*.onnx"))):
        name = os.path.basename(path)
        location = name + "_data"
        model = onnx.load(path, load_external_data=True)
        onnx.save_model(model, path, save_as_external_data=True,
                        all_tensors_to_one_file=True, location=location,
                        size_threshold=1024, convert_attribute=False)
        stale = path + ".data"
        if os.path.exists(stale):
            os.remove(stale)
        size_mb = os.path.getsize(os.path.join(directory, location)) // (1024 * 1024)
        print(f"{name} -> {location} ({size_mb} MB)")


if __name__ == "__main__":
    main()
