# copyright 2003-2012 LOGILAB S.A. (Paris, FRANCE), all rights reserved.
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
"""Run tests.

This will find all modules whose name match a given prefix in the test
directory, and run them. Various command line options provide
additional facilities.

Command line options:

 -v  verbose -- run tests in verbose mode with output to stdout
 -q  quiet   -- don't print anything except if a test fails
 -t  testdir -- directory where the tests will be found
 -x  exclude -- add a test to exclude
 -p  profile -- profiled execution
 -d  dbc     -- enable design-by-contract
 -m  match   -- only run test matching the tag pattern which follow

If no non-option arguments are present, prefixes used are 'test',
'regrtest', 'smoketest' and 'unittest'.

"""


__docformat__ = "restructuredtext en"
# modified copy of some functions from test/regrtest.py from PyXml
# disable camel case warning
# pylint: disable=C0103

import sys
import os
import os.path as osp
import types
import doctest
import inspect
import unittest
import traceback
import tempfile
import warnings

from inspect import isgeneratorfunction, isclass, FrameInfo
from functools import wraps
from itertools import dropwhile
from contextlib import contextmanager

from typing import Any, Iterator, Union, Optional, Callable, Dict, List, Tuple, Generator
from mypy_extensions import NoReturn

from logilab.common import textutils
from logilab.common.debugger import Debugger, colorize_source
from logilab.common.decorators import cached, classproperty


__all__ = ["unittest_main", "find_tests", "nocoverage", "pause_trace"]

DEFAULT_PREFIXES = ("test", "regrtest", "smoketest", "unittest", "func", "validation")

# used by unittest to count the number of relevant levels in the traceback
__unittest = 1


def in_tempdir(callable):
    """A decorator moving the enclosed function inside the tempfile.tempfdir"""

    @wraps(callable)
    def proxy(*args, **kargs):
        old_cwd = os.getcwd()
        os.chdir(tempfile.tempdir)
        try:
            return callable(*args, **kargs)
        finally:
            os.chdir(old_cwd)

    return proxy


def find_tests(testdir, prefixes=DEFAULT_PREFIXES, suffix=".py", excludes=(), remove_suffix=True):
    """
    Return a list of all applicable test modules.
    """
    tests = []
    for name in os.listdir(testdir):
        if not suffix or name.endswith(suffix):
            for prefix in prefixes:
                if name.startswith(prefix):
                    if remove_suffix and name.endswith(suffix):
                        name = name[: -len(suffix)]
                    if name not in excludes:
                        tests.append(name)
    tests.sort()
    return tests


# PostMortem Debug facilities #####
def start_interactive_mode(result):
    """starts an interactive shell so that the user can inspect errors"""
    debuggers = result.debuggers
    descrs = result.error_descrs + result.fail_descrs
    if len(debuggers) == 1:
        # don't ask for test name if there's only one failure
        debuggers[0].start()
    else:
        while True:
            testindex = 0
            print("Choose a test to debug:")
            # order debuggers in the same way than errors were printed
            print("\n".join([f"\t{i} : {descr}" for i, (_, descr) in enumerate(descrs)]))
            print("Type 'exit' (or ^D) to quit")
            print()
            try:
                todebug = input("Enter a test name: ")
                if todebug.strip().lower() == "exit":
                    print()
                    break
                else:
                    try:
                        testindex = int(todebug)
                        debugger = debuggers[descrs[testindex][0]]
                    except (ValueError, IndexError):
                        print(f"ERROR: invalid test number {todebug!r}")
                    else:
                        debugger.start()
            except (EOFError, KeyboardInterrupt):
                print()
                break


# coverage pausing tools #####################################################


@contextmanager
def replace_trace(trace: Optional[Callable] = None) -> Iterator:
    """A context manager that temporary replaces the trace function"""
    oldtrace = sys.gettrace()
    sys.settrace(trace)
    try:
        yield
    finally:
        # specific hack to work around a bug in pycoverage, see
        # https://bitbucket.org/ned/coveragepy/issue/123
        if oldtrace is not None and not callable(oldtrace) and hasattr(oldtrace, "pytrace"):
            oldtrace = oldtrace.pytrace
        sys.settrace(oldtrace)


pause_trace = replace_trace


def nocoverage(func: Callable) -> Callable:
    """Function decorator that pauses tracing functions"""
    if hasattr(func, "uncovered"):
        return func
    # mypy: "Callable[..., Any]" has no attribute "uncovered"
    # dynamic attribute for magic
    func.uncovered = True  # type: ignore

    def not_covered(*args: Any, **kwargs: Any) -> Any:
        with pause_trace():
            return func(*args, **kwargs)

    # mypy: "Callable[[VarArg(Any), KwArg(Any)], NoReturn]" has no attribute "uncovered"
    # dynamic attribute for magic
    not_covered.uncovered = True  # type: ignore
    return not_covered


# test utils ##################################################################


# Add deprecation warnings about new api used by module level fixtures in unittest2
# http://www.voidspace.org.uk/python/articles/unittest2.shtml#setupmodule-and-teardownmodule
class _DebugResult(object):  # simplify import statement among unittest flavors..
    "Used by the TestSuite to hold previous class when running in debug."
    _previousTestClass = None
    _moduleSetUpFailed = False
    shouldStop = False


# backward compatibility: TestSuite might be imported from lgc.testlib
TestSuite = unittest.TestSuite


class keywords(dict):
    """Keyword args (**kwargs) support for generative tests."""


class starargs(tuple):
    """Variable arguments (*args) for generative tests."""

    def __new__(cls, *args):
        return tuple.__new__(cls, args)


unittest_main = unittest.main


class InnerTestSkipped(unittest.SkipTest):
    """raised when a test is skipped"""


def parse_generative_args(params: Tuple[int, ...]) -> Tuple[Union[List[bool], List[int]], Dict]:
    args = []
    varargs = ()
    kwargs: Dict = {}
    flags = 0  # 2 <=> starargs, 4 <=> kwargs
    for param in params:
        if isinstance(param, starargs):
            varargs = param
            if flags:
                raise TypeError("found starargs after keywords !")
            flags |= 2
            args += list(varargs)
        elif isinstance(param, keywords):
            kwargs = param
            if flags & 4:
                raise TypeError("got multiple keywords parameters")
            flags |= 4
        elif flags & 2 or flags & 4:
            raise TypeError("found parameters after kwargs or args")
        else:
            args.append(param)

    return args, kwargs


class InnerTest(tuple):
    def __new__(cls, name, *data):
        instance = tuple.__new__(cls, data)
        instance.name = name
        return instance


class Tags(set):
    """A set of tag able validate an expression"""

    def __init__(self, *tags: str, **kwargs: Any) -> None:
        self.inherit = kwargs.pop("inherit", True)
        if kwargs:
            raise TypeError(f"{kwargs.keys()} are an invalid keyword argument for this function")

        if len(tags) == 1 and not isinstance(tags[0], str):
            tags = tags[0]
        super(Tags, self).__init__(tags)

    def __getitem__(self, key: str) -> bool:
        return key in self

    def match(self, exp: str) -> bool:
        # mypy: Argument 3 to "eval" has incompatible type "Tags";
        # mypy: expected "Optional[Mapping[str, Any]]"
        # I'm really not sure here?
        return eval(exp, {}, self)  # type: ignore

    # mypy: Argument 1 of "__or__" is incompatible with supertype "AbstractSet";
    # mypy: supertype defines the argument type as "AbstractSet[_T]"
    # not sure how to fix this one
    def __or__(self, other: "Tags") -> "Tags":  # type: ignore
        return Tags(*super(Tags, self).__or__(other))


# duplicate definition from unittest2 of the _deprecate decorator
def _deprecate(original_func):
    def deprecated_func(*args, **kwargs):
        warnings.warn(f"Please use {original_func.__name__} instead.", DeprecationWarning, 2)
        return original_func(*args, **kwargs)

    return deprecated_func


class TestCase(unittest.TestCase):
    """A unittest.TestCase extension with some additional methods."""

    maxDiff = None
    tags = Tags()

    def __init__(self, methodName: str = "runTest") -> None:
        super(TestCase, self).__init__(methodName)
        self.__exc_info = sys.exc_info
        self.__testMethodName = self._testMethodName
        self._current_test_descr = None
        self._options_ = None

    @classproperty
    @cached
    def datadir(cls) -> str:  # pylint: disable=E0213
        """helper attribute holding the standard test's data directory

        NOTE: this is a logilab's standard
        """
        mod = sys.modules[cls.__module__]
        return osp.join(osp.dirname(osp.abspath(mod.__file__)), "data")

    # cache it (use a class method to cache on class since TestCase is
    # instantiated for each test run)

    @classmethod
    def datapath(cls, *fname: str) -> str:
        """joins the object's datadir and `fname`"""
        return osp.join(cls.datadir, *fname)

    def set_description(self, descr):
        """sets the current test's description.
        This can be useful for generative tests because it allows to specify
        a description per yield
        """
        self._current_test_descr = descr

    # override default's unittest.py feature
    def shortDescription(self) -> Optional[Any]:
        """override default unittest shortDescription to handle correctly
        generative tests
        """
        if self._current_test_descr is not None:
            return self._current_test_descr
        return super(TestCase, self).shortDescription()

    def quiet_run(self, result: Any, func: Callable, *args: Any, **kwargs: Any) -> bool:
        try:
            func(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except unittest.SkipTest as e:
            if hasattr(result, "addSkip"):
                result.addSkip(self, str(e))
            else:
                warnings.warn(
                    "TestResult has no addSkip method, skips not reported", RuntimeWarning, 2
                )
                result.addSuccess(self)
            return False
        except Exception:
            result.addError(self, self.__exc_info())
            return False
        return True

    def _get_test_method(self) -> Callable:
        """return the test method"""
        return getattr(self, self._testMethodName)

    def optval(self, option, default=None):
        """return the option value or default if the option is not define"""
        return getattr(self._options_, option, default)

    def __call__(self, result=None, runcondition=None, options=None):
        """rewrite TestCase.__call__ to support generative tests
        This is mostly a copy/paste from unittest.py (i.e same
        variable names, same logic, except for the generative tests part)
        """
        if result is None:
            result = self.defaultTestResult()
        self._options_ = options
        # if result.cvg:
        #     result.cvg.start()
        testMethod = self._get_test_method()
        if getattr(self.__class__, "__unittest_skip__", False) or getattr(
            testMethod, "__unittest_skip__", False
        ):
            # If the class or method was skipped.
            try:
                skip_why = getattr(self.__class__, "__unittest_skip_why__", "") or getattr(
                    testMethod, "__unittest_skip_why__", ""
                )
                if hasattr(result, "addSkip"):
                    result.addSkip(self, skip_why)
                else:
                    warnings.warn(
                        "TestResult has no addSkip method, skips not reported", RuntimeWarning, 2
                    )
                    result.addSuccess(self)
            finally:
                result.stopTest(self)
            return
        if runcondition and not runcondition(testMethod):
            return  # test is skipped
        result.startTest(self)
        try:
            if not self.quiet_run(result, self.setUp):
                return
            generative = isgeneratorfunction(testMethod)
            # generative tests
            if generative:
                self._proceed_generative(result, testMethod, runcondition)
            else:
                status = self._proceed(result, testMethod)
                success = status == 0
            if not self.quiet_run(result, self.tearDown):
                return
            if not generative and success:
                result.addSuccess(self)
        finally:
            # if result.cvg:
            #     result.cvg.stop()
            result.stopTest(self)

    def _proceed_generative(
        self, result: Any, testfunc: Callable, runcondition: Callable = None
    ) -> bool:
        # cancel startTest()'s increment
        result.testsRun -= 1
        success = True
        try:
            for params in testfunc():
                if runcondition and not runcondition(testfunc, skipgenerator=False):
                    if not (isinstance(params, InnerTest) and runcondition(params)):
                        continue
                if not isinstance(params, (tuple, list)):
                    params = (params,)
                func = params[0]
                args, kwargs = parse_generative_args(params[1:])
                # increment test counter manually
                result.testsRun += 1
                status = self._proceed(result, func, args, kwargs)
                if status == 0:
                    result.addSuccess(self)
                    success = True
                else:
                    success = False
                    # XXX Don't stop anymore if an error occured
                    # if status == 2:
                    #    result.shouldStop = True
                if result.shouldStop:  # either on error or on exitfirst + error
                    break
        except self.failureException:
            result.addFailure(self, self.__exc_info())
            success = False
        except unittest.SkipTest as e:
            result.addSkip(self, e)
        except Exception:
            # if an error occurs between two yield
            result.addError(self, self.__exc_info())
            success = False
        return success

    def _proceed(
        self,
        result: Any,
        testfunc: Callable,
        args: Union[List[bool], List[int], Tuple[()]] = (),
        kwargs: Optional[Dict] = None,
    ) -> int:
        """proceed the actual test
        returns 0 on success, 1 on failure, 2 on error

        Note: addSuccess can't be called here because we have to wait
        for tearDown to be successfully executed to declare the test as
        successful
        """
        kwargs = kwargs or {}
        try:
            testfunc(*args, **kwargs)
        except self.failureException:
            result.addFailure(self, self.__exc_info())
            return 1
        except KeyboardInterrupt:
            raise
        except InnerTestSkipped as e:
            result.addSkip(self, e)
            return 1
        except unittest.SkipTest as e:
            result.addSkip(self, e)
            return 0
        except Exception:
            result.addError(self, self.__exc_info())
            return 2
        return 0

    def innerSkip(self, msg: str = None) -> NoReturn:
        """mark a generative test as skipped for the <msg> reason"""
        msg = msg or "test was skipped"
        raise InnerTestSkipped(msg)

    if sys.version_info >= (3, 2):
        assertItemsEqual = unittest.TestCase.assertCountEqual
    else:
        assertCountEqual = unittest.TestCase.assertItemsEqual


class SkippedSuite(unittest.TestSuite):
    def test(self):
        """just there to trigger test execution"""
        self.skipped_test("doctest module has no DocTestSuite class")


class DocTestFinder(doctest.DocTestFinder):
    def __init__(self, *args, **kwargs):
        self.skipped = kwargs.pop("skipped", ())
        doctest.DocTestFinder.__init__(self, *args, **kwargs)

    def _get_test(self, obj, name, module, globs, source_lines):
        """override default _get_test method to be able to skip tests
        according to skipped attribute's value
        """
        if getattr(obj, "__name__", "") in self.skipped:
            return None
        return doctest.DocTestFinder._get_test(self, obj, name, module, globs, source_lines)


class MockConnection:
    """fake DB-API 2.0 connexion AND cursor (i.e. cursor() return self)"""

    def __init__(self, results):
        self.received = []
        self.states = []
        self.results = results

    def cursor(self):
        """Mock cursor method"""
        return self

    def execute(self, query, args=None):
        """Mock execute method"""
        self.received.append((query, args))

    def fetchone(self):
        """Mock fetchone method"""
        return self.results[0]

    def fetchall(self):
        """Mock fetchall method"""
        return self.results

    def commit(self):
        """Mock commiy method"""
        self.states.append(("commit", len(self.received)))

    def rollback(self):
        """Mock rollback method"""
        self.states.append(("rollback", len(self.received)))

    def close(self):
        """Mock close method"""


# mypy error: Name 'Mock' is not defined
# dynamic class created by this class
def mock_object(**params: Any) -> "Mock":  # type: ignore # noqa
    """creates an object using params to set attributes
    >>> option = mock_object(verbose=False, index=range(5))
    >>> option.verbose
    False
    >>> option.index
    [0, 1, 2, 3, 4]
    """
    return type("Mock", (), params)()


def create_files(paths: List[str], chroot: str) -> None:
    """Creates directories and files found in <path>.

    :param paths: list of relative paths to files or directories
    :param chroot: the root directory in which paths will be created

    >>> from os.path import isdir, isfile
    >>> isdir('/tmp/a')
    False
    >>> create_files(['a/b/foo.py', 'a/b/c/', 'a/b/c/d/e.py'], '/tmp')
    >>> isdir('/tmp/a')
    True
    >>> isdir('/tmp/a/b/c')
    True
    >>> isfile('/tmp/a/b/c/d/e.py')
    True
    >>> isfile('/tmp/a/b/foo.py')
    True
    """
    dirs, files = set(), set()
    for path in paths:
        path = osp.join(chroot, path)
        filename = osp.basename(path)
        # path is a directory path
        if filename == "":
            dirs.add(path)
        # path is a filename path
        else:
            dirs.add(osp.dirname(path))
            files.add(path)
    for dirpath in dirs:
        if not osp.isdir(dirpath):
            os.makedirs(dirpath)
    for filepath in files:
        open(filepath, "w").close()


class AttrObject:  # XXX cf mock_object
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def tag(*args: str, **kwargs: Any) -> Callable:
    """descriptor adding tag to a function"""

    def desc(func: Callable) -> Callable:
        assert not hasattr(func, "tags")
        # mypy: "Callable[..., Any]" has no attribute "tags"
        # dynamic magic attribute
        func.tags = Tags(*args, **kwargs)  # type: ignore
        return func

    return desc


def require_version(version: str) -> Callable:
    """Compare version of python interpreter to the given one. Skip the test
    if older.
    """

    def check_require_version(f: Callable) -> Callable:
        version_elements = version.split(".")
        try:
            compare = tuple([int(v) for v in version_elements])
        except ValueError:
            raise ValueError(f"{version} is not a correct version : should be X.Y[.Z].")
        current = sys.version_info[:3]
        if current < compare:

            def new_f(self, *args, **kwargs):
                self.skipTest(
                    "Need at least %s version of python. Current version is %s."
                    % (version, ".".join([str(element) for element in current]))
                )

            new_f.__name__ = f.__name__
            return new_f
        else:
            return f

    return check_require_version


def require_module(module: str) -> Callable:
    """Check if the given module is loaded. Skip the test if not."""

    def check_require_module(f: Callable) -> Callable:
        try:
            __import__(module)
            return f
        except ImportError:

            def new_f(self, *args, **kwargs):
                self.skipTest(f"{module} can not be imported.")

            new_f.__name__ = f.__name__
            return new_f

    return check_require_module


class SkipAwareTextTestRunner(unittest.TextTestRunner):
    def __init__(
        self,
        stream=sys.stderr,
        verbosity=1,
        exitfirst=False,
        pdbmode=False,
        cvg=None,
        test_pattern=None,
        skipped_patterns=(),
        colorize=False,
        batchmode=False,
        options=None,
    ):
        super(SkipAwareTextTestRunner, self).__init__(stream=stream, verbosity=verbosity)
        self.exitfirst = exitfirst
        self.pdbmode = pdbmode
        self.cvg = cvg
        self.test_pattern = test_pattern
        self.skipped_patterns = skipped_patterns
        self.colorize = colorize
        self.batchmode = batchmode
        self.options = options

    def does_match_tags(self, test: Callable) -> bool:
        if self.options is not None:
            tags_pattern = getattr(self.options, "tags_pattern", None)
            if tags_pattern is not None:
                tags = getattr(test, "tags", Tags())
                if tags.inherit and isinstance(test, types.MethodType):
                    tags = tags | getattr(test.__self__.__class__, "tags", Tags())
                return tags.match(tags_pattern)
        return True  # no pattern

    def _makeResult(self) -> "SkipAwareTestResult":
        return SkipAwareTestResult(
            self.stream,
            self.descriptions,
            self.verbosity,
            self.exitfirst,
            self.pdbmode,
            self.cvg,
            self.colorize,
        )


class SkipAwareTestResult(unittest._TextTestResult):
    def __init__(
        self,
        stream,
        descriptions: bool,
        verbosity: int,
        exitfirst: bool = False,
        pdbmode: bool = False,
        cvg: Optional[Any] = None,
        colorize: bool = False,
    ) -> None:
        super(SkipAwareTestResult, self).__init__(stream, descriptions, verbosity)
        self.skipped: List[Tuple[Any, Any]] = []
        self.debuggers: List = []
        self.fail_descrs: List = []
        self.error_descrs: List = []
        self.exitfirst = exitfirst
        self.pdbmode = pdbmode
        self.cvg = cvg
        self.colorize = colorize
        self.pdbclass = Debugger
        self.verbose = verbosity > 1

    def descrs_for(self, flavour: str) -> List[Tuple[int, str]]:
        return getattr(self, f"{flavour.lower()}_descrs")

    def _create_pdb(self, test_descr: str, flavour: str) -> None:
        self.descrs_for(flavour).append((len(self.debuggers), test_descr))
        if self.pdbmode:
            self.debuggers.append(self.pdbclass(sys.exc_info()[2]))

    def _iter_valid_frames(self, frames: List[FrameInfo]) -> Generator[FrameInfo, Any, None]:
        """only consider non-testlib frames when formatting  traceback"""

        def invalid(fi):
            return osp.abspath(fi[1]) in (lgc_testlib, std_testlib)

        lgc_testlib = osp.abspath(__file__)
        std_testlib = osp.abspath(unittest.__file__)

        for frameinfo in dropwhile(invalid, frames):
            yield frameinfo

    def _exc_info_to_string(self, err, test):
        """Converts a sys.exc_info()-style tuple of values into a string.

        This method is overridden here because we want to colorize
        lines if --color is passed, and display local variables if
        --verbose is passed
        """
        exctype, exc, tb = err
        output = ["Traceback (most recent call last)"]
        frames = inspect.getinnerframes(tb)
        colorize = self.colorize
        frames = enumerate(self._iter_valid_frames(frames))
        for index, (frame, filename, lineno, funcname, ctx, ctxindex) in frames:
            filename = osp.abspath(filename)
            if ctx is None:  # pyc files or C extensions for instance
                source = "<no source available>"
            else:
                source = "".join(ctx)
            if colorize:
                filename = textutils.colorize_ansi(filename, "magenta")
                source = colorize_source(source)
            output.append(f'  File "{filename}", line {lineno}, in {funcname}')
            output.append(f"    {source.strip()}")
            if self.verbose:
                output.append(f"{dir(frame)!r} == {test.__module__!r}")
                output.append("")
                output.append("    " + " local variables ".center(66, "-"))
                for varname, value in sorted(frame.f_locals.items()):
                    output.append(f"    {varname}: {value!r}")
                    if varname == "self":  # special handy processing for self
                        for varname, value in sorted(vars(value).items()):
                            output.append(f"      self.{varname}: {value!r}")
                output.append("    " + "-" * 66)
                output.append("")
        output.append("".join(traceback.format_exception_only(exctype, exc)))
        return "\n".join(output)

    def addError(self, test, err):
        """err ->  (exc_type, exc, tcbk)"""
        exc_type, exc, _ = err
        if isinstance(exc, unittest.SkipTest):
            assert exc_type == unittest.SkipTest
            self.addSkip(test, exc)
        else:
            if self.exitfirst:
                self.shouldStop = True
            descr = self.getDescription(test)
            super(SkipAwareTestResult, self).addError(test, err)
            self._create_pdb(descr, "error")

    def addFailure(self, test, err):
        if self.exitfirst:
            self.shouldStop = True
        descr = self.getDescription(test)
        super(SkipAwareTestResult, self).addFailure(test, err)
        self._create_pdb(descr, "fail")

    def addSkip(self, test, reason):
        self.skipped.append((test, reason))
        if self.showAll:
            self.stream.writeln("SKIPPED")
        elif self.dots:
            self.stream.write("S")

    def printErrors(self) -> None:
        super(SkipAwareTestResult, self).printErrors()
        self.printSkippedList()

    def printSkippedList(self) -> None:
        # format (test, err) compatible with unittest2
        for test, err in self.skipped:
            descr = self.getDescription(test)
            self.stream.writeln(self.separator1)
            self.stream.writeln(f"{'SKIPPED'}: {descr}")
            self.stream.writeln(f"\t{err}")

    def printErrorList(self, flavour, errors):
        for (_, descr), (test, err) in zip(self.descrs_for(flavour), errors):
            self.stream.writeln(self.separator1)
            self.stream.writeln(f"{flavour}: {descr}")
            self.stream.writeln(self.separator2)
            self.stream.writeln(err)
            self.stream.writeln("no stdout".center(len(self.separator2)))
            self.stream.writeln("no stderr".center(len(self.separator2)))


class NonStrictTestLoader(unittest.TestLoader):
    """
    Overrides default testloader to be able to omit classname when
    specifying tests to run on command line.

    For example, if the file test_foo.py contains ::

        class FooTC(TestCase):
            def test_foo1(self): # ...
            def test_foo2(self): # ...
            def test_bar1(self): # ...

        class BarTC(TestCase):
            def test_bar2(self): # ...

    'python test_foo.py' will run the 3 tests in FooTC
    'python test_foo.py FooTC' will run the 3 tests in FooTC
    'python test_foo.py test_foo' will run test_foo1 and test_foo2
    'python test_foo.py test_foo1' will run test_foo1
    'python test_foo.py test_bar' will run FooTC.test_bar1 and BarTC.test_bar2
    """

    def __init__(self) -> None:
        self.skipped_patterns = ()

    # some magic here to accept empty list by extending
    # and to provide callable capability
    def loadTestsFromNames(self, names: List[str], module: type = None) -> TestSuite:
        suites = []
        for name in names:
            suites.extend(self.loadTestsFromName(name, module))
        return self.suiteClass(suites)

    def _collect_tests(self, module: type) -> Dict[str, Tuple[type, List[str]]]:
        tests = {}
        for obj in vars(module).values():
            if isclass(obj) and issubclass(obj, unittest.TestCase):
                classname = obj.__name__
                if classname[0] == "_" or self._this_is_skipped(classname):
                    continue
                methodnames = []
                # obj is a TestCase class
                for attrname in dir(obj):
                    if attrname.startswith(self.testMethodPrefix):
                        attr = getattr(obj, attrname)
                        if callable(attr):
                            methodnames.append(attrname)
                # keep track of class (obj) for convenience
                tests[classname] = (obj, methodnames)
        return tests

    def loadTestsFromSuite(self, module, suitename):
        try:
            suite = getattr(module, suitename)()
        except AttributeError:
            return []
        assert hasattr(suite, "_tests"), "%s.%s is not a valid TestSuite" % (
            module.__name__,
            suitename,
        )
        # python2.3 does not implement __iter__ on suites, we need to return
        # _tests explicitly
        return suite._tests

    def loadTestsFromName(self, name, module=None):
        parts = name.split(".")
        if module is None or len(parts) > 2:
            # let the base class do its job here
            return [super(NonStrictTestLoader, self).loadTestsFromName(name)]
        tests = self._collect_tests(module)
        collected = []
        if len(parts) == 1:
            pattern = parts[0]
            if callable(getattr(module, pattern, None)) and pattern not in tests:
                # consider it as a suite
                return self.loadTestsFromSuite(module, pattern)
            if pattern in tests:
                # case python unittest_foo.py MyTestTC
                klass, methodnames = tests[pattern]
                for methodname in methodnames:
                    collected = [klass(methodname) for methodname in methodnames]
            else:
                # case python unittest_foo.py something
                for klass, methodnames in tests.values():
                    # skip methodname if matched by skipped_patterns
                    for skip_pattern in self.skipped_patterns:
                        methodnames = [
                            methodname
                            for methodname in methodnames
                            if skip_pattern not in methodname
                        ]
                    collected += [
                        klass(methodname) for methodname in methodnames if pattern in methodname
                    ]
        elif len(parts) == 2:
            # case "MyClass.test_1"
            classname, pattern = parts
            klass, methodnames = tests.get(classname, (None, []))
            for methodname in methodnames:
                collected = [
                    klass(methodname) for methodname in methodnames if pattern in methodname
                ]
        return collected

    def _this_is_skipped(self, testedname: str) -> bool:
        # mypy: Need type annotation for 'pat'
        # doc doesn't say how to that in list comprehension
        return any([(pat in testedname) for pat in self.skipped_patterns])  # type: ignore

    def getTestCaseNames(self, testCaseClass: type) -> List[str]:
        """Return a sorted sequence of method names found within testCaseClass"""
        is_skipped = self._this_is_skipped
        classname = testCaseClass.__name__
        if classname[0] == "_" or is_skipped(classname):
            return []
        testnames = super(NonStrictTestLoader, self).getTestCaseNames(testCaseClass)
        return [testname for testname in testnames if not is_skipped(testname)]


# The 2 functions below are modified versions of the TestSuite.run method
# that is provided with unittest2 for python 2.6, in unittest2/suite.py
# It is used to monkeypatch the original implementation to support
# extra runcondition and options arguments (see in testlib.py)


def _ts_wrapped_run(
    self: Any,
    result: SkipAwareTestResult,
    debug: bool = False,
    runcondition: Callable = None,
    options: Optional[Any] = None,
) -> SkipAwareTestResult:
    for test in self:
        if result.shouldStop:
            break
        if unittest.suite._isnotsuite(test):
            self._tearDownPreviousClass(test, result)
            self._handleModuleFixture(test, result)
            self._handleClassSetUp(test, result)
            result._previousTestClass = test.__class__
            if getattr(test.__class__, "_classSetupFailed", False) or getattr(
                result, "_moduleSetUpFailed", False
            ):
                continue

        # --- modifications to deal with _wrapped_run ---
        # original code is:
        #
        # if not debug:
        #     test(result)
        # else:
        #     test.debug()
        if hasattr(test, "_wrapped_run"):
            try:
                test._wrapped_run(result, debug, runcondition=runcondition, options=options)
            except TypeError:
                test._wrapped_run(result, debug)
        elif not debug:
            try:
                test(result, runcondition, options)
            except TypeError:
                test(result)
        else:
            test.debug()
        # --- end of modifications to deal with _wrapped_run ---
    return result


def _ts_run(  # noqa
    self: Any,
    result: SkipAwareTestResult,
    debug: bool = False,
    runcondition: Callable = None,
    options: Optional[Any] = None,
) -> SkipAwareTestResult:
    topLevel = False
    if getattr(result, "_testRunEntered", False) is False:
        result._testRunEntered = topLevel = True

    self._wrapped_run(result, debug, runcondition, options)

    if topLevel:
        self._tearDownPreviousClass(None, result)
        self._handleModuleTearDown(result)
        result._testRunEntered = False
    return result


# monkeypatch unittest and doctest (ouch !)
unittest._TextTestResult = SkipAwareTestResult
unittest.TextTestRunner = SkipAwareTextTestRunner
unittest.TestLoader = NonStrictTestLoader

unittest.FunctionTestCase.__bases__ = (TestCase,)
unittest.TestSuite.run = _ts_run

unittest.TestSuite._wrapped_run = _ts_wrapped_run
