import os
import requests


def resolve_model_path() -> str:
    """Download (once) and return the local phonikud ONNX model path.

    Passed to scriptconv phonemizers as their lazy ``phonikud_model``
    resolver — scriptconv itself never downloads.
    """
    base_path = os.path.expanduser("~/.local/share/phonikud")
    fname = PhonikudDiacritizer.dl_url.split("/")[-1]
    model = f"{base_path}/{fname}"
    if not os.path.isfile(model):
        os.makedirs(base_path, exist_ok=True)
        data = requests.get(PhonikudDiacritizer.dl_url).content
        with open(model, "wb") as f:
            f.write(data)
    return model


class PhonikudDiacritizer:
    dl_url = "https://huggingface.co/thewh1teagle/phonikud-onnx/resolve/main/phonikud-1.0.int8.onnx"

    def __init__(self):

        base_path = os.path.expanduser("~/.local/share/phonikud")
        fname = self.dl_url.split("/")[-1]
        model = f"{base_path}/{fname}"
        if not os.path.isfile(model):
            os.makedirs(base_path, exist_ok=True)
            # TODO - streaming download
            data = requests.get(self.dl_url).content
            with open(model, "wb") as f:
                f.write(data)

        from phonikud_onnx import Phonikud
        self.phonikud = Phonikud(model)

    def diacritize(self, text: str) -> str:
        return self.phonikud.add_diacritics(text)