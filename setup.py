from setuptools import setup

setup(
    name='phoonnx',
    version='0.0.0',
    packages=['phoonnx',
              'phoonnx.phonemizers',
              'phoonnx.thirdparty',
              'phoonnx.thirdparty.kog2p',
              'phoonnx.thirdparty.tashkeel',
              'phoonnx.thirdparty.mantoq',
              'phoonnx.thirdparty.mantoq.buck',
              'phoonnx.thirdparty.mantoq.pyarabic',
              'phoonnx_train',
              'phoonnx_train.vits',
              'phoonnx_train.vits.monotonic_align',
              'phoonnx_train.norm_audio',
              'phoonnx_train.norm_audio.models'],
    include_package_data=True,
    url='https://github.com/TigreGotico/phoonnx',
    license='',
    author='JarbasAi',
    author_email='jarbasai@mailfence.com',
    description=''
)
