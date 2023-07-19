from pathlib import Path
import subprocess, sys, os
from setuptools import setup, find_packages
from subprocess import check_call

# from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install


DS9CADET_text = """CADET
*
bind c
DS9CADET CADET -x $xpa_method  $param(CADET); -dec $decompose -shift $shift -th1 $threshold1 -th2 $threshold2  | $text
#DS9CADET CADET -x $xpa_method  $param(CADET); -dec $decompose -shift $shift -b $bootstrap -n $boot_n -th1 $threshold1 -th2 $threshold2  | $text

CADET
*
menu
DS9CADET CADET -x $xpa_method  $param(CADET); -dec $decompose -shift $shift -th1 $threshold1 -th2 $threshold2  | $text
#DS9CADET CADET -x $xpa_method  $param(CADET); -dec $decompose -shift $shift -b $bootstrap -n $boot_n -th1 $threshold1 -th2 $threshold2  | $text

param CADET
shift checkbox {Shift} 0 {Shift the input region by +/- 1 pixel (increases execution time 8 times).}
# bootstrap checkbox {Bootstrap} 0 {Boostrap individual counts of the input image (increases execution time N times).}
# boot_n entry {Bootstrap N} 1 {Number of bootstrap iterations per single rotation-shifting configuration.}
decompose checkbox {Decompose} 1 {Decompose raw cavity prediction into individual cavities.}
threshold1 entry {Threshold1} 0.5 {Volume calibrating threshold (only applied if Decompose).}
threshold2 entry {Threshold2} 0.9 {TP/FP calibrating threshold (only applied if Decompose).}
endparam
"""


def ReplaceStringInFile(path, string1, string2, path2=None):
    """Replaces string in a txt file"""
    fin = open(path, "rt")
    data = fin.read()
    data = data.replace(string1, string2)
    fin.close()
    if path2 is not None:
        path = path2
        if os.path.exists(path):
            os.remove(path)
        fin = open(path, "x")
    else:
        fin = open(path, "wt")
    fin.write(data)
    fin.close()
    return

def LoadDS9CADET():
    """Load the plugin in DS9 parameter file
    """

    ds9 = os.popen("whereis ds9").read()

    if ds9 != "":
        # try:
        if 1:
            home = os.environ['HOME']
            DS9CADET_path = f"{home}/.ds9/DS9CADET.ds9.ans"

            os.system(f"mkdir -p {home}/.ds9")

            with open(DS9CADET_path, "w") as f:
                f.write(DS9CADET_text)

            version = os.popen("ds9 -version").read().replace("\n","").replace("ds9 ", "")            
            pref = f"{home}/.ds9/ds9.{version}.prf"

            if os.path.isfile(pref):
                print("DS9 parameter file found")
                print(pref)
                ReplaceStringInFile(path=pref, string1="user4 {}", string2="user4 {%s}" % (DS9CADET_path))
            else:
                print("DS9 parameter file not found. Creating new one.")
                with open(pref, "w") as f:
                    f.write("global panalysis\narray set panalysis { user2 {} autoload 1 user3 {} log 0 user4 {} user " + DS9CADET_path + " }")
                os.chmod(pref, 0o644)
        # except:
        #     print(f"Encountered an error while opening DS9. The preferances file (e.g. ~.ds9/ds9.8.3.prf) might be corrupted.")
    else:
        print("DS9 not found. If you want to use CADET as DS9 plugin, please install DS9 first.")


class PostDevelopCommand(develop):
    def run(self):
        for package in requirements:
            pip_install(package)
        develop.run(self)
        LoadDS9CADET()

class PostInstallCommand(install):
    def run(self):
        for package in requirements:
            pip_install(package)
        install.run(self)
        LoadDS9CADET()


def pip_install(package_name):
    try: check_call([sys.executable, "-m", "pip", "install", package_name])
    except: pass


requirements = ["numpy >=1.8",
                "matplotlib", # >= 3.1.1",
                "astropy >=1.3",
                "scipy >=0.14",
                "pyds9 >= 1.8.1",
                "scikit-learn >= 1.2",
                "tensorflow >= 2.8"]

entry_points = {}
entry_points["console_scripts"] = ["DS9CADET = pycadet.DS9CADET:main"]

data = {
    "pycadet": [
        "DS9CADET",
        "*.hdf5",
    ]
}

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

MAJOR = "0"
MINOR = "1"
MICRO = "3"
version = "%s.%s.%s" % (MAJOR, MINOR, MICRO)

setup(
    name="pycadet",
    version=version,
    author="Tomas Plsek",
    author_email="plsek@physics.muni.cz",
    description="Cavity Detection Tool",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/tomasplsek/CADET",
    # install_requires=requirements,
    entry_points=entry_points,
    cmdclass={"install": PostInstallCommand, "develop": PostDevelopCommand},
    packages=find_packages(exclude=("docs", "training_testing", "examples",)),
    package_data=data,
    include_package_data=False,
)