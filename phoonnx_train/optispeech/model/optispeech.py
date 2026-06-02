import torch

from phoonnx_train.optispeech.dataset.feature_extractors import CommonFeatureExtractor
from phoonnx_train.optispeech.dataset.feature_extractors.pitch_extractors import DIOPitchExtractor
from phoonnx_train.optispeech.model.base_lightning_module import BaseLightningModule
from phoonnx_train.optispeech.model.generator import OptiSpeechGenerator
from phoonnx_train.optispeech.model.generator.modules import ConvNeXtBackbone
from phoonnx_train.optispeech.model.generator.modules import DurationPredictor
from phoonnx_train.optispeech.model.generator.modules import EnergyPredictor
from phoonnx_train.optispeech.model.generator.modules import PitchPredictor
from phoonnx_train.optispeech.model.generator.modules import TextEmbedding
from phoonnx_train.optispeech.model.vocoder.wavenext import WaveNeXt
from phoonnx_train.optispeech.model.vocoder.wavenext.disc import VocosDiscriminator
from phoonnx_train.optispeech.text import TextProcessor
from phoonnx_train.optispeech.values import InferenceInputs, InferenceOutputs


def load():
    from transformers import get_cosine_schedule_with_warmup

    return OptiSpeechGenerator(
        dim=256,
        generator=OptiSpeechGenerator(
            segment_size=64,
            energy_predictor=EnergyPredictor(
                num_layers=2,
                intermediate_size=384,
                kernel_size=3,
                dropout=0.5,
                embed_kernel_size=9,
                embed_dropout=0.5,
                conv_layer_class=torch.nn.Conv1d
            ),
            pitch_predictor=PitchPredictor(
                num_layers=5,
                intermediate_size=256,
                kernel_size=5,
                dropout=0.5,
                embed_kernel_size=9,
                embed_dropout=0.2,
                conv_layer_class=torch.nn.Conv1d
            ),
            duration_predictor=DurationPredictor(
                num_layers=2,
                intermediate_dim=384,
                kernel_size=3,
                dropout=0.1,
                conv_layer_class=torch.nn.Conv1d
            ),
            encoder=ConvNeXtBackbone(
                intermediate_dim=1024,
                num_layers=4,
                drop_path=0.2
            ),
            decoder=ConvNeXtBackbone(
                intermediate_dim=1024,
                num_layers=4,
                drop_path=0.2
            ),
            text_embedding=TextEmbedding(
                n_vocab=250,
                dropout=0.1,
                padding_idx=0,
                max_source_positions=2000
            )
        ),
        vocoder=WaveNeXt(
            dim=384,
            intermediate_dim=1152,
            num_layers=8,
            drop_path=0.1
        ),
        loss_coeffs={
            "lambda_align": 5.0,
            "lambda_duration": 1.0,
            "lambda_pitch": 1.0,
            "lambda_energy": 1.0,
        },
        inference_args={
            "d_factor": 1.1,
            "p_factor": 1.6,
            "e_factor": 1.2
        },
        data_args={
            # "name": "ljspeech",
            "num_speakers": 1,
            "batch_size": 128,
            "num_workers": 8,
            "pin_memory": True,

            "text_processor": TextProcessor(
                tokenizer="ipa",
                add_blank="",
                add_bos_eos="",
                normalize_text=True,
                languages=["en-us"]
            ),
            "feature_extractor": CommonFeatureExtractor(
                sample_rate=24000,
                n_feats=100,
                n_fft=1024,
                hop_length=256,
                win_length=1024,
                f_min=80,
                f_max=8000,
                center=True,
                pitch_extractor=DIOPitchExtractor(
                    batch_size=2048,
                    interpolate=True
                ),
                preemphasis_filter_coef=None,
                lowpass_freq=None,
                highpass_freq=None,
                loudness_norm_target_db=-24,
                trim_silence=False,
                trim_silence_args={
                    "silence_threshold": 0.1,
                    "silence_samples_per_chunk": 720,
                    "silence_keep_chunks_before": 1,
                    "silence_keep_chunks_after": 1,
                }
            ),
            # "data_statistics":{ },
        },
        train_args={
            "cache_generator_outputs": False,
            "gradient_clip_val": 10,
            "gradient_accumulate_batches": None,
            "pretraining_steps": 1000,
            "evaluate_periodicity": True,
            "evaluate_utmos": True,
            "evaluate_pesq": True,
        },
        optimizer=torch.optim.AdamW(lr=2e-4,
                                    betas=(0.8, 0.99),
                                    weight_decay=1e-2),
        scheduler=get_cosine_schedule_with_warmup
    )


class OptiSpeech(BaseLightningModule):
    def __init__(
            self,
            dim: int,  # 256
            generator: OptiSpeechGenerator,
            vocoder: WaveNeXt,
            discriminator: VocosDiscriminator,
            train_args,
            data_args,
            inference_args,
            optimizer: torch.optim.AdamW = None,
            scheduler=None,  # transformers.get_cosine_schedule_with_warmup
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Sanity checks
        if (train_args.gradient_accumulate_batches is not None) and (train_args.gradient_accumulate_batches <= 0):
            raise ValueError("gradient_accumulate_batches should be a positive number")

        if data_args.num_speakers < 1:
            raise ValueError("num_speakers should be a positive integer >= 1")

        self.train_args = train_args
        self.data_args = data_args
        self.inference_args = inference_args

        self.text_processor = self.data_args.text_processor

        self.num_speakers = data_args.num_speakers
        self.sample_rate = data_args.feature_extractor.sample_rate
        self.hop_length = data_args.feature_extractor.hop_length

        # GAN training requires this
        self.automatic_optimization = False

        self.generator = generator(
            dim=dim,
            vocoder=vocoder,
            feature_extractor=data_args.feature_extractor,
            data_statistics=data_args.data_statistics,
            num_speakers=self.data_args.num_speakers,
            num_languages=self.text_processor.num_languages,
        )
        self.discriminator = discriminator(feature_extractor=data_args.feature_extractor)

    @torch.inference_mode()
    def synthesise(self, inputs: InferenceInputs) -> InferenceOutputs:
        inputs = inputs.as_torch()
        inputs = inputs.to(self.device)
        synth_outputs = self.generator.synthesise(
            x=inputs.x,
            x_lengths=inputs.x_lengths.to("cpu"),
            sids=inputs.sids,
            lids=inputs.lids,
            d_factor=inputs.d_factor,
            p_factor=inputs.p_factor,
            e_factor=inputs.e_factor
        )
        return InferenceOutputs(
            wav=synth_outputs["wav"],
            wav_lengths=synth_outputs["wav_lengths"],
            durations=synth_outputs["durations"],
            pitch=synth_outputs["pitch"],
            energy=synth_outputs["energy"],
            latency=synth_outputs["latency"],
            rtf=synth_outputs["rtf"],
            am_rtf=synth_outputs["am_rtf"],
            v_rtf=synth_outputs["v_rtf"],
        )

    def prepare_input(
            self,
            text: str,
            *,
            language: str | None = None,
            speaker: str | int | None = None,
            d_factor: float = None,
            p_factor: float = None,
            e_factor: float = None,
            split_sentences: bool = True,
    ) -> InferenceInputs:
        """
        Convenient helper.

        Args:
            text (str): input text
            language (str|None): language of input text
            speaker (int|str|None): speaker name
            d_factor (float|None): scaling value for duration
            p_factor (float|None): scaling value for pitch
            e_factor (float|None): scaling value for energy
            split_sentences (bool): split text into sentences (each sentence is an element in the batch)

        Returns:
            InferenceInputs
        """
        languages = self.text_processor.languages
        if language is None:
            language = languages[0]
        if self.num_speakers > 1:
            if speaker is None:
                sid = 0
            elif type(speaker) is str:
                try:
                    sid = self.speakers.index(speaker)
                except IndexError:
                    raise ValueError(f"A speaker with the given name `{speaker}` was not found in speaker list")
            elif type(speaker) is int:
                sid = speaker
        else:
            sid = None
        if self.text_processor.is_multi_language:
            try:
                lid = languages.index(language)
            except IndexError:
                raise ValueError(f"A language with the given name `{language}` was not found in language list")
        else:
            lid = None

        input_ids, clean_text = self.text_processor(text, lang=language, split_sentences=split_sentences)
        if split_sentences:
            lengths = [len(phids) for phids in input_ids]
        else:
            lengths = [len(input_ids)]
            input_ids = [input_ids]

        sids = [sid] * len(input_ids) if sid is not None else None
        lids = [lid] * len(input_ids) if lid is not None else None

        inputs = InferenceInputs.from_ids_and_lengths(
            ids=input_ids,
            lengths=lengths,
            clean_text=clean_text,
            sids=sids,
            lids=lids,
            d_factor=d_factor or self.inference_args.d_factor,
            p_factor=p_factor or self.inference_args.p_factor,
            e_factor=e_factor or self.inference_args.e_factor
        )
        inputs = inputs.as_torch()
        inputs = inputs.to(self.device)
        return inputs
