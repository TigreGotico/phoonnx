import os

from setuptools import setup

BASEDIR = os.path.abspath(os.path.dirname(__file__))


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


def required(requirements_file):
    """ Read requirements file and remove comments and empty lines. """
    with open(os.path.join(os.path.dirname(__file__), requirements_file),
              'r') as f:
        requirements = f.read().splitlines()
        return [pkg for pkg in requirements
                if pkg.strip() and not pkg.startswith("#")]


def get_version():
    """ Find the version of the package"""
    version_file = os.path.join(BASEDIR, 'phoonnx', 'version.py')
    major, minor, build, alpha = (None, None, None, None)
    with open(version_file) as f:
        for line in f:
            if 'VERSION_MAJOR' in line:
                major = line.split('=')[1].strip()
            elif 'VERSION_MINOR' in line:
                minor = line.split('=')[1].strip()
            elif 'VERSION_BUILD' in line:
                build = line.split('=')[1].strip()
            elif 'VERSION_ALPHA' in line:
                alpha = line.split('=')[1].strip()

            if ((major and minor and build and alpha) or
                    '# END_VERSION_BLOCK' in line):
                break
    version = f"{major}.{minor}.{build}"
    if alpha and int(alpha) > 0:
        version += f"a{alpha}"
    return version


extra_files = package_files('phoonnx')


setup(
    name='phoonnx',
    version=get_version(),
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
    package_data={'': extra_files},
    install_requires=required('requirements/requirements.txt'),
    extras_require={
        x.split(".")[0]: required(f'requirements/{x}')
        for x in os.listdir(f"{os.path.dirname(__file__)}/requirements")
        if x.endswith(".txt") and x != "requirements.txt"

    },
    url='https://github.com/TigreGotico/phoonnx',
    license='',
    author='JarbasAi',
    author_email='jarbasai@mailfence.com',
    description=''
)
