import codecs
import os
import sys
import six
import pytest
import unittest
import ast
import codegen
import astpretty


DEBUG_OUT = True
EXCEPTION_MSG_CHECKING = True

# Standard python syntax

py_for_else = """\
for x in range(10):
    break
else:
    y = 2
"""

py_for_range_arg1 = """\
for x in range(10):
    break
"""
cpp_for_range_arg1 = """\
for (int x=0;x<10;x++){
    break;
}
"""

py_for_range_arg2 = """\
for x in range(2, 11):
    break
"""
cpp_for_range_arg2 = """\
for (int x=2;x<11;x++){
    break;
}
"""

py_for_range_arg3 = """\
for x in range(3, 12, 4):
    break
"""
cpp_for_range_arg3 = """\
for (int x=3;x<12;x+=4){
    break;
}
"""

py_for_unsupported = """\
for x in something:
    break
"""

py_while_else = """\
while True:
    break
else:
    y = 2
"""

py_while = """\
while True:
    break
"""
cpp_while = """\
while (true){
    break;
}
"""

py_try = """\
try:
    1 / 0
except Exception as e:
    pass
"""

py_async_func = """\
async def async_function():
    pass
"""

py_class_decorator = """\
@f1(arg)
@f2
class Foo: pass
"""

py_elif1 = """\
if cond1:
    break
elif cond2:
    break
else:
    break
"""

cpp_elif1 = """\
if (cond1){
    break;
}
else if (cond2){
    break;
}
else{
    break;
}
"""

py_elif2 = """\
if cond1:
    break
elif cond2:
    break
"""

cpp_elif2 = """\
if (cond1){
    break;
}
else if (cond2){
    break;
}
"""

py_with_simple = """\
with f():
    suite1
"""

py_with_as = """\
with f() as x:
    suite1
"""


py_async_function_def = """\
async def f():
    suite1
"""

py_async_for = """\
async for _ in reader:
    suite1
"""

py_async_with = """\
async with g():
    suite1
"""

py_async_with_as = """\
async with g() as x:
    suite1
"""

# FGPU functionality

py_fgpu_for_msg_input = """\
for msg in message_in: 
    f = msg.getVariableFloat("f")
    fa4 = msg.getVariableFloatArray4("fa4")
"""

cpp_fgpu_for_msg_input = """\
for (const auto& msg : FLAMEGPU->message_in){
    auto f = msg.getVariable<float>("f");
    auto fa4 = msg.getVariable<float, 4>("fa4");
}
"""

class CodeGenTest(unittest.TestCase):


    def _checkExpected(self, source, expected=None):
        source = source.strip()
        tree = ast.parse(source)
        if DEBUG_OUT:
            astpretty.pprint(tree)
        code = codegen.codegen(tree)
        # remove new lines
        code = code.strip()
        if expected:
            expected = expected.strip()
        else:
            expected = source
        if DEBUG_OUT:
            print(f"Expected: {expected}")
            print(f"Output  : {code}")
        assert expected == code
        
    def _checkException(self, source, exception_str):
        with pytest.raises(codegen.CodeGenException) as e:
            tree = ast.parse(source.strip())
            # code generate
            code = codegen.codegen(tree)
        if EXCEPTION_MSG_CHECKING:
            assert exception_str in str(e.value)
        

    def test_del_statement(self):
        self._checkException("del x, y, z", "Deletion not supported")

    def test_shifts(self):
        self._checkExpected("45 << 2", "(45 << 2);")
        self._checkExpected("13 >> 7", "(13 >> 7);")

    def test_for_else(self):
        self._checkException(py_for_else, "For else not supported")
        
    def test_for_range(self):
        # use of range based for loops (calling range with different number of arguments)
        self._checkExpected(py_for_range_arg1, cpp_for_range_arg1)
        self._checkExpected(py_for_range_arg2, cpp_for_range_arg2)
        self._checkExpected(py_for_range_arg3, cpp_for_range_arg3)   
        # check that non range function loops are rejected
        self._checkException(py_for_unsupported, "Range based for loops only support")
       
    def test_while_else(self):
        self._checkException(py_while_else, "While else not supported")
        
    def test_while(self):
        self._checkExpected(py_while, cpp_while)

    def test_unary_parens(self):
        self._checkExpected("(-1)**7", "pow((-1), 7);")
        self._checkExpected("-1.**8", "(-pow(1.0, 8));")
        self._checkExpected("not True or False", "((!true) || false);")
        self._checkExpected("True or not False", "(true || (!false));")

    def test_integer_parens(self):
        self._checkException("3 .__abs__()", "Unsupported") # should resolve to unsupported function call syntax

    def test_huge_float(self):
        self._checkExpected("1e1000", "inf;")
        #self._checkExpected("-1e1000")
        #self._checkExpected("1e1000j")
        #self._checkExpected("-1e1000j")

    def test_min_int30(self):
        self._checkExpected(str(-2**31), "(-2147483648);")
        self._checkExpected(str(-2**63), "(-9223372036854775808);")

    def test_negative_zero(self):
        self._checkExpected("-0", "(-0);")
        self._checkExpected("-(0)", "(-0);")
        self._checkExpected("-0b0", "(-0);")
        self._checkExpected("-(0b0)", "(-0);")
        self._checkExpected("-0o0", "(-0);")
        self._checkExpected("-(0o0)", "(-0);")
        self._checkExpected("-0x0", "(-0);")
        self._checkExpected("-(0x0)", "(-0);")

    def test_lambda_parentheses(self):
        self._checkException("(lambda: int)()", "Lambda is not supported")

    def test_chained_comparisons(self):
        self._checkExpected("1 < 4 <= 5", "1 < 4 <= 5;")
        self._checkExpected("a is b is c is not d", "a == b == c != d;")

    def test_function_arguments(self):
        # only flame gpu functions or device functions are supported
        self._checkException("def f(): pass", "Function definitions require a")

    def test_relative_import(self):
        self._checkException("from . import fred", "Importing of modules not supported")

    def test_import_many(self):
        self._checkException("import fred, other", "Importing of modules not supported")

    def test_nonlocal(self):
        self._checkException("nonlocal x", "Use of 'nonlocal' not supported")

    def test_exceptions(self):
        self._checkException("raise Error", "Exception raising not supported")
        self._checkException(py_try, "Exceptions not supported")

    def test_bytes(self):
        self._checkException("b'123'", "Byte strings not supported")

    def test_strings(self):
        self._checkException('f"{value}"', "not supported")

    def test_set_literal(self):
        self._checkException("{'a', 'b', 'c'}", "Sets not supported")

    def test_comprehension(self):
        self._checkException("{x for x in range(5)}", "Set comprehension not supported")
        self._checkException("{x: x*x for x in range(10)}", "Dictionary comprehension not supported")

    def test_dict_with_unpacking(self):
        self._checkException("{**x}", "Dictionaries not supported")
        self._checkException("{a: b, **x}", "Dictionaries not supported")

    def test_async_comp_and_gen_in_async_function(self):
        self._checkException(py_async_func, "Async functions not supported")

    def test_async_comprehension(self):
        self._checkException("{i async for i in aiter() if i % 2}", "Set comprehension not supported")
        self._checkException("[i async for i in aiter() if i % 2]", "List comprehension not supported")
        self._checkException("{i: -i async for i in aiter() if i % 2}", "Dictionary comprehension not supported")

    def test_async_generator_expression(self):
        self._checkException("(i ** 2 async for i in agen())", "Generator expressions not supported")
        self._checkException("(i - 1 async for i in agen() if i % 2)", "Generator expressions not supported")

    def test_class(self):
        self._checkException("class Foo: pass", "Class definitions not supported")
        self._checkException(py_class_decorator, "Class definitions not supported")


    def test_elifs(self):
        self._checkExpected(py_elif1, cpp_elif1)
        self._checkExpected(py_elif2, cpp_elif2)


    def test_starred_assignment(self):
        self._checkException("a, *b = seq", "Assignment to complex expressions not supported")

    def test_variable_annotation(self):
        self._checkExpected("a: int", "int a;")
        self._checkExpected("a: int = 0", "int a = 0;")
        self._checkExpected("a: int = None", "int a = 0;")
        self._checkException("some_list: List[int]", "Not a supported type")
        self._checkException("some_list: List[int] = []", "Not a supported type")
        self._checkException("t: Tuple[int, ...] = (1, 2, 3)", "Not a supported type")

    def test_with(self):
        self._checkException(py_with_simple, "With not supported")
        self._checkException(py_with_as, "With not supported")
        self._checkException(py_async_with, "Async with not supported")
        self._checkException(py_async_with_as, "Async with not supported")

    def test_async_function_def(self):
        self._checkException(py_async_function_def, "Async functions not supported")

    def test_async_for(self):
        self._checkException(py_async_for, "Async for not supported")



# FLAME GPU specific functionality

    def test_fgpu_supported_types(self):
        self._checkExpected("a: numpy.byte", "char a;")
        self._checkExpected("a: numpy.ubyte", "unsigned char a;"),
        self._checkExpected("a: numpy.short", "short a;")
        self._checkExpected("a: numpy.ushort", "unsigned short a;")
        self._checkExpected("a: numpy.intc", "int a;")
        self._checkExpected("a: numpy.uintc", "unsigned int a;")
        self._checkExpected("a: numpy.uint", "unisgned int a;")
        self._checkExpected("a: numpy.longlong", "long long a;")
        self._checkExpected("a: numpy.ulonglong", "unsigned long long a;")
        self._checkExpected("a: numpy.half", "half a;")
        self._checkExpected("a: numpy.single", "float a;")
        self._checkExpected("a: numpy.double", "double a;")
        self._checkExpected("a: numpy.longdouble", "long double a;")
        self._checkExpected("a: numpy.bool_", "bool a;")
        self._checkExpected("a: numpy.bool8", "bool a;")
        # sized aliases
        self._checkExpected("a: numpy.int_", "long a;")
        self._checkExpected("a: numpy.int8", "int8_t a;"),
        self._checkExpected("a: numpy.int16", "int16_t a;")
        self._checkExpected("a: numpy.int32", "int32_t a;")
        self._checkExpected("a: numpy.int64", "int64_t a;")
        self._checkExpected("a: numpy.intp", "intptr_t a;")
        self._checkExpected("a: numpy.uint_", "long a;")
        self._checkExpected("a: numpy.uint8", "uint8_t a;")
        self._checkExpected("a: numpy.uint16", "uint16_t a;")
        self._checkExpected("a: numpy.uint32", "uint32_t a;")
        self._checkExpected("a: numpy.uint64", "uint64_t a;")
        self._checkExpected("a: numpy.uintp", "uintptr_t a;")
        self._checkExpected("a: numpy.float_", "float a;")
        self._checkExpected("a: numpy.float16", "half a;")
        self._checkExpected("a: numpy.float32", "float a;")
        self._checkExpected("a: numpy.float64", "double a;")
        # check unsupported
        self._checkException("a: numpy.unsupported", "Not a supported numpy type")
        
        
    def test_msg_input(self):
        self._checkExpected(py_fgpu_for_msg_input, cpp_fgpu_for_msg_input)
    
    @unittest.skip
    def test_function_arguments(self):
        self._checkExpected("def f(): pass", "")
        self._checkExpected("def f(a): pass")
        self._checkExpected("def f(b = 2): pass")
        self._checkExpected("def f(a, b): pass")
        self._checkExpected("def f(a, b = 2): pass")
        self._checkExpected("def f(a = 5, b = 2): pass")
        self._checkExpected("def f(*args, **kwargs): pass")
        if six.PY3:
            self._checkExpected("def f(*, a = 1, b = 2): pass")
            self._checkExpected("def f(*, a = 1, b): pass")
            self._checkExpected("def f(*, a, b = 2): pass")
            self._checkExpected("def f(a, b = None, *, c, **kwds): pass")
            self._checkExpected("def f(a=2, *args, c=5, d, **kwds): pass")
            
    @unittest.skip
    def test_annotations(self):
        self._checkExpected("def f(a : int): pass")
        self._checkExpected("def f(a: int = 5): pass")
        self._checkExpected("def f(*args: [int]): pass")
        self._checkExpected("def f(**kwargs: dict): pass")
        self._checkExpected("def f() -> None: pass")
    
