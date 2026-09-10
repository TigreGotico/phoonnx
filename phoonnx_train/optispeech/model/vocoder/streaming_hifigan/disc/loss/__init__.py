from phoonnx_train.optispeech.model.vocoder.streaming_hifigan.disc.base_vocoder_disc import BaseVocoderDiscriminator, \
    LossOutput

from phoonnx_train.optispeech.model.vocoder.streaming_hifigan.disc.loss.adversarial_loss import *  # NOQA
from phoonnx_train.optispeech.model.vocoder.streaming_hifigan.disc.loss.feat_match_loss import *  # NOQA
from phoonnx_train.optispeech.model.vocoder.streaming_hifigan.disc.loss.mel_loss import *  # NOQA
from phoonnx_train.optispeech.model.vocoder.streaming_hifigan.disc.loss.stft_loss import *  # NOQA
from phoonnx_train.optispeech.model.vocoder.streaming_hifigan.disc.loss.waveform_loss import *  # NOQA


class HiFiGANDiscriminator(BaseVocoderDiscriminator):

    def __init__(self):
        ...

    def forward_disc(self, wav, wav_hat) -> LossOutput:
        """Calculate discriminator's loss for training batch"""

    def forward_gen(self, wav, wav_hat) -> LossOutput:
        """Calculate adversarial loss for training batch"""

    def forward_val(self, wav, wav_hat) -> LossOutput:
        """Calculate loss for validation batch."""
