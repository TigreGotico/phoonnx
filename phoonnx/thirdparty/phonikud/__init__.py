import os
import requests


class PhonikudDiacritizer:
    dl_url = "https://huggingface.co/thewh1teagle/phonikud-onnx/resolve/main/phonikud-1.0.int8.onnx"

    @classmethod
    def download(cls) -> str:
        """Ensure the phonikud ONNX model exists locally and return its path.

        scriptconv never downloads the Hebrew model itself, so phoonnx stays
        responsible for fetching it; callers pass the returned path to
        ``scriptconv.diacritics.diacritize(..., phonikud_model=<path>)``.
        """
        base_path = os.path.expanduser("~/.local/share/phonikud")
        fname = cls.dl_url.split("/")[-1]
        model = f"{base_path}/{fname}"
        if not os.path.isfile(model):
            os.makedirs(base_path, exist_ok=True)
            # TODO - streaming download
            data = requests.get(cls.dl_url).content
            with open(model, "wb") as f:
                f.write(data)
        return model

    def __init__(self):
        model = self.download()
        from phonikud_onnx import Phonikud
        self.phonikud = Phonikud(model)

    def diacritize(self, text: str) -> str:
        return self.phonikud.add_diacritics(text)