from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="manipulathor",
        packages=find_packages(),
        version="0.0.1",
        install_requires=[
            "allenact==0.2.2",
            "allenact_plugins[ithor]==0.2.2",
            "setuptools",
        ],
    )
