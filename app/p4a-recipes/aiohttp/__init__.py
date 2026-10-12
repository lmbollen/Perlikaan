"""aiohttp as pure Python.

python-for-android's own aiohttp recipe pins 3.8.3, whose pre-generated C
does not compile against Python 3.14. edge-tts only needs aiohttp's
pure-Python implementation, which every release provides when
AIOHTTP_NO_EXTENSIONS is set, so this recipe installs a current release
without its C accelerators. Its dependencies (multidict, yarl, frozenlist,
propcache, aiosignal, attrs, aiohappyeyeballs) are listed in buildozer.spec
and installed by python-for-android itself; build.sh makes them pure Python
too.
"""
from pythonforandroid.recipe import PythonRecipe


class AiohttpRecipe(PythonRecipe):
    name = "aiohttp"
    version = "3.14.5"
    url = "https://pypi.python.org/packages/source/a/aiohttp/aiohttp-{version}.tar.gz"
    depends = ["setuptools"]
    call_hostpython_via_targetpython = False
    # pip would otherwise install aiohttp's dependencies here too, built for
    # the build machine rather than the phone.
    setup_extra_args = ["--no-deps"]

    def get_recipe_env(self, arch):
        env = super().get_recipe_env(arch)
        env["AIOHTTP_NO_EXTENSIONS"] = "1"
        return env


recipe = AiohttpRecipe()
