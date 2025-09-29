import os
import requests


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