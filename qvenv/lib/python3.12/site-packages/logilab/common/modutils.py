# copyright 2003-2013 LOGILAB S.A. (Paris, FRANCE), all rights reserved.
# contact http://www.logilab.fr/ -- mailto:contact@logilab.fr
#
# This file is part of logilab-common.
#
# logilab-common is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 2.1 of the License, or (at your option) any
# later version.
#
# logilab-common is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with logilab-common.  If not, see <http://www.gnu.org/licenses/>.
"""Python modules manipulation utility functions.
"""

__docformat__ = "restructuredtext en"

import sys
import os
from os.path import (
    join,
    abspath,
    exists,
    expanduser,
    normcase,
    realpath,
)
from typing import Dict, List, Optional, Sequence

from importlib import import_module

from logilab.common import STD_BLACKLIST, _handle_blacklist
from logilab.common.deprecation import callable_deprecated


class LazyObject:
    """
    This class allows to lazyly declare a object (most likely only a callable
    according to the code) from a module without importing it.

    The import will be triggered when the user tries to access attributes of
    the object/callable or call it.

    Trying to set or delete attributes of the wrapped object/callable will not
    works as expected.
    """

    def __init__(self, module, obj):
        self.module = module
        self.obj = obj
        self._imported = None

    def _getobj(self):
        if self._imported is None:
            self._imported = getattr(import_module(self.module), self.obj)
        return self._imported

    def __getattribute__(self, attr):
        try:
            return super(LazyObject, self).__getattribute__(attr)
        except AttributeError:
            return getattr(self._getobj(), attr)

    def __call__(self, *args, **kwargs):
        return self._getobj()(*args, **kwargs)


def _check_init(path: str, mod_path: List[str]) -> bool:
    """check there are some __init__.py all along the way"""

    def _has_init(directory: str) -> Optional[str]:
        """if the given directory has a valid __init__ file, return its path,
        else return None
        """
        mod_or_pack = join(directory, "__init__")

        for ext in ("py", "pyc", "pyo"):
            if exists(mod_or_pack + "." + ext):
                return mod_or_pack + "." + ext

        return None

    def _has_dirs(directory: str) -> bool:
        for file in os.listdir(directory):
            if os.path.isdir(os.path.join(directory, path)):
                return True

        return False

    for part in mod_path:
        path = join(path, part)
        if not _has_init(path) and not _has_dirs(path):
            return False
    return True


@callable_deprecated(
    "you should avoid using modpath_from_file(), it doesn't play well with symlinks and "
    "sys.meta_path and you should use python standard loaders"
)
def modpath_from_file(filename: str, extrapath: Optional[Dict[str, str]] = None) -> List[str]:
    """DEPRECATED: doesn't play well with symlinks and sys.meta_path

    Given a file path return the corresponding splitted module's name
    (i.e name of a module or package splitted on '.')

    :type filename: str
    :param filename: file's path for which we want the module's name

    :type extrapath: dict
    :param extrapath:
      optional extra search path, with path as key and package name for the path
      as value. This is usually useful to handle package splitted in multiple
      directories using __path__ trick.


    :raise ImportError:
      if the corresponding module's name has not been found

    :rtype: list(str)
    :return: the corresponding splitted module's name
    """

    def _canonicalize_path(path: str) -> str:
        return realpath(expanduser((path)))

    def _is_in_a_valid_module(directory_or_file: str) -> bool:
        """
        Try to emulate a reverse version of the new rule of PEP 420 to
        determine if a file is in a valid module.

        https://peps.python.org/pep-0420/

        To quote it:

        > During import processing, the import machinery will continue to
        > iterate over each directory in the parent path as it does in Python
        > 3.2. While looking for a module or package named “foo”, for each
        > directory in the parent path:
        >
        >  * If <directory>/foo/__init__.py is found, a regular package is imported and returned.
        >  * If not, but <directory>/foo.{py,pyc,so,pyd} is found, a module is imported and
        >    returned. The exact list of extension varies by platform and whether the -O flag is
        >    specified. The list here is representative.
        >  * If not, but <directory>/foo is found and is a directory, it is recorded and the scan
        >    continues with the next directory in the parent path.
        >  * Otherwise the scan continues with the next directory in the parent path.
        >
        > If the scan completes without returning a module or package, and at least one
        > directory was recorded, then a namespace package is created. The new namespace
        > package:
        >
        >  * Has a __path__ attribute set to an iterable of the path strings that were
        >    found and recorded during the scan.
        >  * Does not have a __file__ attribute.
        """
        # XXX to quote documentation: The exact list of extension varies by
        # platform and whether the -O flag is specified.
        # So this code is not great at all
        python_extensions = ("py", "pyc", "pyo", "so", "pyd")

        directory = directory_or_file
        if not os.path.isdir(directory_or_file):
            # <directory>/foo.{py,pyc,so,pyd} situation
            if os.path.exists(directory_or_file) and directory_or_file.endswith(
                tuple("." + extension for extension in python_extensions)
            ):
                return True

            directory = os.path.split(directory_or_file)[0]

        mod_or_pack = join(directory, "__init__")

        for ext in python_extensions:
            # <directory>/foo/__init__.py case
            if exists(mod_or_pack + "." + ext):
                return True

        return False

    filename = _canonicalize_path(filename)
    base = os.path.splitext(filename)[0]

    if extrapath is not None:
        for path_ in map(_canonicalize_path, extrapath):
            path = abspath(path_)
            if path and normcase(base[: len(path)]) == normcase(path):
                if _is_in_a_valid_module(filename):
                    submodpath = [pkg for pkg in base[len(path) :].split(os.sep) if pkg]
                    return extrapath[path_].split(".") + submodpath

    for path in map(_canonicalize_path, sys.path):
        if path and normcase(base).startswith(path):
            modpath = [pkg for pkg in base[len(path) :].split(os.sep) if pkg]
            if _is_in_a_valid_module(filename):
                return modpath

    raise ImportError(
        "Unable to find module for %s in:\n* %s"
        % (filename, "\n* ".join(sys.path + list(extrapath.keys() if extrapath else [])))
    )


def get_module_files(src_directory: str, blacklist: Sequence[str] = STD_BLACKLIST) -> List[str]:
    """given a package directory return a list of all available python
    module's files in the package and its subpackages

    :type src_directory: str
    :param src_directory:
      path of the directory corresponding to the package

    :type blacklist: list or tuple
    :param blacklist:
      optional list of files or directory to ignore, default to the value of
      `logilab.common.STD_BLACKLIST`

    :rtype: list
    :return:
      the list of all available python module's files in the package and
      its subpackages
    """
    files = []
    for directory, dirnames, filenames in os.walk(src_directory):
        _handle_blacklist(blacklist, dirnames, filenames)
        # check for __init__.py
        if "__init__.py" not in filenames:
            dirnames[:] = ()
            continue
        for filename in filenames:
            if filename.endswith((".py", ".so", ".pyd", ".pyw")):
                src = join(directory, filename)
                files.append(src)
    return files


def cleanup_sys_modules(directories):
    """remove submodules of `directories` from `sys.modules`"""
    cleaned = []
    for modname, module in list(sys.modules.items()):
        modfile = getattr(module, "__file__", None)
        if modfile:
            for directory in directories:
                if modfile.startswith(directory):
                    cleaned.append(modname)
                    del sys.modules[modname]
                    break
    return cleaned


def clean_sys_modules(names):
    """remove submodules starting with name from `names` from `sys.modules`"""
    cleaned = set()
    for modname in list(sys.modules):
        for name in names:
            if modname.startswith(name):
                del sys.modules[modname]
                cleaned.add(modname)
                break
    return cleaned
