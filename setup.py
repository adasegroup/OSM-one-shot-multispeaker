from setuptools import setup, find_packages
try:
    # pip >=20
    from pip._internal.network.session import PipSession
    from pip._internal.req import parse_requirements
except ImportError:
    try:
        # 10.0.0 <= pip <= 19.3.1
        from pip._internal.download import PipSession
        from pip._internal.req import parse_requirements
    except ImportError:
        # pip <= 9.0.3
        from pip.download import PipSession
        from pip.req import parse_requirements


def load_requirements(file_name):
    requirements = parse_requirements(file_name, session=PipSession())
    try:
        return [str(item.req) for item in requirements]
    except:
        return [str(item.requirement) for item in requirements]


setup(
    name="osms",
    version="1.0.0",
    description="OSM: One-Shot Multi-speaker",
    author="Nikolay Kozyrskiy, Gleb Balitskiy",
    author_email="nikolay.kozyrskiy@skoltech.ru",
    url="https://github.com/adasegroup/OSM-one-shot-multispeaker",
    packages=find_packages(),
    license="MIT",
    install_requires=load_requirements("requirements.txt")
)
