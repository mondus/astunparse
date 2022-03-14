"Usage: unparse.py <path to source file>"
from __future__ import print_function, unicode_literals
import six
import sys
import ast
import os
import tokenize
from six import StringIO

# Large float and imaginary literals get turned into infinities in the AST.
# We unparse those infinities to INFSTR.
INFSTR = "1e" + repr(sys.float_info.max_10_exp + 1)

def interleave(inter, f, seq):
    """Call f on each item in seq, calling inter() in between.
    """
    seq = iter(seq)
    try:
        f(next(seq))
    except StopIteration:
        pass
    else:
        for x in seq:
            inter()
            f(x)

class Unparser:
    """Methods in this class recursively traverse an AST and
    output source code for the abstract syntax; original formatting
    is disregarded. """

    def __init__(self, tree, file = sys.stdout):
        """Unparser(tree, file=sys.stdout) -> None.
         Print the source for tree to file."""
        self.f = file
        self.future_imports = []
        self._indent = 0
        # dict of locals used to determine if variable already exists in assignments
        self._locals = ["FLAMEGPU"]
        self._device_functions = [] # not actually checked anyf unction call is allowed for now
        self._message_iterator_var = None;
        self.dispatch(tree)
        print("", file=self.f)
        self.f.flush()


    def fill(self, text = ""):
        "Indent a piece of text, according to the current indentation level"
        self.f.write("\n"+"    "*self._indent + text)

    def write(self, text):
        "Append a piece of text to the current line."
        self.f.write(six.text_type(text))

    def enter(self):
        "Print '{', and increase the indentation."
        self.write("{")
        self._indent += 1

    def leave(self):
        "Decrease the indentation level and Print '}'"
        self._indent -= 1
        self.fill("}")

    def dispatch(self, tree):
        "Dispatcher function, dispatching tree type T to method _T."
        if isinstance(tree, list):
            for t in tree:
                self.dispatch(t)
            return
        meth = getattr(self, "_"+tree.__class__.__name__)
        meth(tree)
        
    def RaiseWarning(self, tree, str):
        print(f"Warning ({tree.lineno}, {tree.col_offset}): {str}")
        
    def RaiseError(self, tree, str):
        print(f"Error ({tree.lineno}, {tree.col_offset}): {str}")


    ### Validation of format functions
    
    def dispatchFGPUFunctionArgs(self, tree):
        if len(tree.args) is not 2:
            self.RaiseError("Expected two FLAME GPU function arguments (input message and output message)")
        MessageTypes = ["MessageNone", "MessageBruteForce"]
        # input message
        if not tree.args[0].annotation:
            self.RaiseError(tree.args[0], "Message input requires a supported type annotation")
        if tree.args[0].annotation.id not in MessageTypes:
            self.RaiseError(tree.args[0], "Message input type annotation not a supported message type")
        self._input_message_var = tree.args[0].arg  # store the message input variable name
        self.dispatch(tree.args[0].annotation)
        self.write(", ")
        # output message
        if not tree.args[1].annotation:
            self.RaiseError(tree.args[1], "Message output requires a supported type annotation")
        if tree.args[1].annotation.id not in MessageTypes:
            self.RaiseError(tree.args[1], "Message output type annotation not a supported message type")
        self._output_message_var = tree.args[1].arg  # store the message output variable name
        self.dispatch(tree.args[1].annotation)
    
    def dispatchType(self, tree):
        """
        There is a limited set of types and formats of type description supported. Types can be either;
        1) A python built in type of int or float, or
        2) A subset of numpy types prefixed with either numpy or np. e.g. np.int16
        This function translates and a catches unsupported types but does not translate a function call (i.e. cast)
        """
        if isinstance(tree, ast.Name):
            if tree.id not in self.basic_arg_types:
                self.RaiseError(arg, "Not a supported type")
            self.write(tree.id)
        elif isinstance(tree, ast.Attribute):
            if not isinstance(tree.value, ast.Name) :
                self.RaiseError(arg, "Not a supported type")
            if not (tree.value.id == "numpy" or tree.value.id == "np"):
                self.RaiseError(arg, "Not a supported type")
            if tree.attr not in self.numpytypes:
                self.RaiseError(arg, "Not a supported numpy type")
            self.write(self.numpytypes[tree.attr])

    
    def dispatchFGPUDeviceFunctionArgs(self, tree):
        # input message
        first = True
        annotation = None
        for arg in tree.args:
            # ensure that there is a type annotation
            if not arg.annotation:
                self.RaiseError(arg, "Device function argument requires type annotation")
            # comma if not first
            if not first:
                self.write(", ")
            self.dispatchType(arg.annotation)
            self.write(f" {arg.arg}")   
            first = False    
    
    def dispatchMessageLoop(self, tree):
        self.fill("for (const auto& ")
        self.dispatch(tree.target)
        self.write(" : FLAMEGPU->")
        self.dispatch(tree.iter)
        self.write(")")
        self._message_iterator_var = tree.target.id
        self.enter()
        self.dispatch(tree.body)
        self.leave()
        self._message_iterator_var = None
    
    # argument
    def _FGPU2arg(self, t):
        self.write(t.arg)
        if t.annotation:
            self.write(": ")
            self.dispatch(t.annotation)    
    
    # represents built in functions
    pythonbuiltins = ["abs", "float", "int"]
    
    # basic types
    basic_arg_types = ['float', 'int']
    
    # supported math constansts    
    mathconsts = {"pi": "M_PI",
                  "e": "M_E",
                  "inf": "INFINITY",
                  "nan": "NAN",
                  }
    
    # support for most numpy types except complex numbers and float>64bit
    numpytypes = {"byte": "signed char",
                  "byte": "unsigned char",
                  "short": "short",
                  "ushort": "unsigned short",
                  "intc": "int",
                  "uintc": "unsigned int",
                  "uint": "unisgned int",
                  "longlong": "long long",
                  "ulonglong": "unsigned long long",
                  "half": "half",       # cuda supported
                  "single": "float",
                  "double": "double",
                  "longdouble": "long double",
                  "bool_": "bool",
                  "bool8": "bool",
                  # sized aliases
                  "int_": "long",
                  "int8": "int8_t",
                  "int16": "int16_t",
                  "int32": "int32_t",
                  "int64": "int64_t",
                  "intp": "intptr_t",
                  "uint_": "long",
                  "uint8": "uint8_t",
                  "uint16": "uint16_t",
                  "uint32": "uint32_t",
                  "uint64": "uint64_t",
                  "uintp": "uintptr_t",
                  "float_": "float",
                  "float16": "half",
                  "float32": "float",
                  "float64": "double"
                  }

    ############### Unparsing methods ######################
    # There should be one method per concrete grammar type #
    # Constructors should be grouped by sum type. Ideally, #
    # this would follow the order in the grammar, but      #
    # currently doesn't.                                   #
    ########################################################

    def _Module(self, tree):
        for stmt in tree.body:
            self.dispatch(stmt)

    def _Interactive(self, tree):
        for stmt in tree.body:
            self.dispatch(stmt)

    def _Expression(self, tree):
        self.dispatch(tree.body)

    # stmt
    def _Expr(self, tree):
        """
        Same as a standard python expression but ends with semicolon
        """
        self.fill()
        self.dispatch(tree.value)
        self.write(";")

    def _NamedExpr(self, tree):
        """
        No such concept in C++. Standard assignment can be used in any location.
        """
        self.write("(")
        self.dispatch(tree.target)
        self.write(" = ")
        self.dispatch(tree.value)
        self.write(")")

    def _Import(self, t):
        self.RaiseError(t, "Importing of modules not supported")

    def _ImportFrom(self, t):
        self.RaiseError(t, "Importing of modules not supported")

    def _Assign(self, t):
        """
        Assignment will use the auto type to define a variable at first use else will perform standard assignment.
        Note: There is no ability to create `const` variables unless this is inferred from the assignment expression.
        Multiple assignment is supported by cpp but not in the translator neither is assignment to complex expressions which are valid python syntax.
        """
        if len(t.targets) > 1:
            self.RaiseError(t, "Assignment to multiple targets not supported")
        if not isinstance(t.targets[0], ast.Name):
            self.RaiseError(t, "Assignment to complex expressions not supported")
        self.fill()
        # check if target exists in locals
        if t.targets[0].id not in self._locals :
            self.write("auto ")
            self._locals.append(t.targets[0].id)
        self.dispatch(t.targets[0])
        self.write(" = ")
        self.dispatch(t.value)
        self.write(";")

    def _AugAssign(self, t):
        """
        Similar to assignment in terms of restrictions. E.g. Allow only single named variable assignments.
        Also requires the named variable to already exist in scope.
        """
        if not isinstance(t.target, ast.Name):
            self.RaiseError(t, "Augmented assignment to complex expressions not supported")
        # check if target exists in locals
        if t.target.id not in self._locals :
            self.RaiseError(t, "Augmented assignment not permitted on variables not already assigned previously")
        self.fill()
        self.dispatch(t.target)
        self.write(" "+self.binop[t.op.__class__.__name__]+"= ")
        self.dispatch(t.value)
        self.write(";")

    def _AnnAssign(self, t):
        self.RaiseError(t, "Annotated Assignment not supported")

    def _Return(self, t):
        """
        Standard cpp like return with semicolon.
        """
        self.fill("return")
        if t.value:
            self.write(" ")
            self.dispatch(t.value)
        self.write(";")

    def _Pass(self, t):
        self.fill("pass;")

    def _Break(self, t):
        self.fill("break;")

    def _Continue(self, t):
        self.fill("continue;")

    def _Delete(self, t):
        self.RaiseError(t, "Deletion not supported")

    def _Assert(self, t):
        """
        cassert does exist but probably not required in FGPU functions and unclear if supported by jitfy
        """
        self.RaiseError(t, "Assert not supported")

    def _Exec(self, t):
        self.RaiseError(t, "Exec not supported")

    def _Print(self, t):
        """
        This is old school python printing so no need to support
        """
        self.RaiseError(t, "Print not supported")
        
    def _Global(self, t):
        self.RaiseError(t, "Use of 'global' not supported")

    def _Nonlocal(self, t):
        self.RaiseError(t, "Use of 'nonlocal' not supported")

    def _Await(self, t):
        self.RaiseError(t, "Await not supported")

    def _Yield(self, t):
        self.RaiseError(t, "Yield not supported")

    def _YieldFrom(self, t):
        self.RaiseError(t, "Yield from not supported")

    def _Raise(self, t):
        """
        Exceptions are obviously supported in cpp but not in CUDA device code
        """
        self.RaiseError(t, "Exception raising not supported")

    def _Try(self, t):
        self.RaiseError(t, "Exceptions not supported")

    def _TryExcept(self, t):
        self.RaiseError(t, "Exceptions not supported")

    def _TryFinally(self, t):
        self.RaiseError(t, "Exceptions not supported")

    def _ExceptHandler(self, t):
        self.RaiseError(t, "Exceptions not supported")

    def _ClassDef(self, t):
        self.RaiseError(t, "Class definitions not supported")

    def _FunctionDef(self, t):
        """
        Checks the decorators of the function definition much must be either 'flamegpu_agent_function' or 'flamegpu_device_function'.
        Each is then processed in a different way using a specific dispatcher.
        Function calls are actually checked and only permitted (or user defined) function calls are supported.
        """
        self.write("\n")
        # check decorators
        if len(t.decorator_list) is not 1 or not isinstance(t.decorator_list[0], ast.Name):
            self.RaiseError(t, "Function definitions require a single FLAMEGPU decorator of either 'flamegpu_agent_function' or 'flamegpu_device_function'")       
        # FLAMEGPU_AGENT_FUNCTION
        if t.decorator_list[0].id == 'flamegpu_agent_function' :
            if getattr(t, "returns", False):
                self.RaiseWarning(t, "Function definition return type not supported on 'flamegpu_agent_function'")
            self.fill(f"FLAMEGPU_AGENT_FUNCTION({t.name}, ")
            self.dispatchFGPUFunctionArgs(t.args)
            self.write(")")
        # FLAMEGPU_DEVICE_FUNCTION
        elif t.decorator_list[0].id == 'flamegpu_device_function' :
            self.fill(f"FLAMEGPU_DEVICE_FUNCTION ")
            if t.returns:
                self.dispatchType(t.returns)
            else:
                self.write("void")
            self.write(f" {t.name}(")
            self.dispatchFGPUDeviceFunctionArgs(t.args)
            self.write(")")
            # add to list of defined functions that can be called
            self._device_functions.append(t.name)
        else:
            self.RaiseError(t, "Functions= definition uses an unsupported decorator. Must use either 'flamegpu_agent_function' or 'flamegpu_device_function'")
        self.enter()
        self.dispatch(t.body)
        self.leave()

    def _AsyncFunctionDef(self, t):
        self.RaiseError(t, "Async function snot supported")

    def _For(self, t):
        """
        Two type for for loop are supported. Either;
        1) Message for loop in which case the format requires a iterator using the named FLAMEGPU function argument of 'message_in'
        2) A range based for loop with 1 to 3 arguments which is converted into a c style loop
        """
        # if message loop then process differently
        if isinstance(t.iter, ast.Name):
            if t.iter.id == "message_in":
                self.dispatchMessageLoop(t)
            else:
                self.RaiseError(t, "Range based for loops only support message iteration using 'message_in' iterator")
        # allow calls but only to range function
        elif isinstance(t.iter, ast.Call):
            if isinstance(t.iter.func, ast.Name):
                if t.iter.func.id == "range":
                    # switch on different uses of range based on number of arguments
                    if len(t.iter.args) == 1:
                        self.fill(f"for (int ")
                        self.dispatch(t.target)
                        self.write("=0;i<")
                        self.dispatch(t.iter.args[0])
                        self.write(";i++)")
                    elif len(t.iter.args) == 2:
                        self.fill(f"for (int ")
                        self.dispatch(t.target)
                        self.write("=")
                        self.dispatch(t.iter.args[0])
                        seld.write(";i<")
                        self.dispatch(t.iter.args[1])
                        self.write(";i++)")
                    elif len(t.iter.args) == 3:
                        self.fill(f"for (int ")
                        self.dispatch(t.target)
                        self.write("=")
                        self.dispatch(t.iter.args[0])
                        seld.write(";i<")
                        self.dispatch(t.iter.args[1])
                        self.write(";i+=")
                        self.dispatch(t.iter.args[2])
                        self.write(")")
                    else:
                        self.RaiseError(t, "Range based for loops requires use of 'range' function with arguments and not keywords")
                    self.enter()
                    self.dispatch(t.body)
                    self.leave()
                else:
                    self.RaiseError(t, "Range based for loops only support calls to the 'range' function")
            else:
                self.RaiseError(t, "Range based for loops only support message iteration or use of 'range'")
        else:
            self.RaiseError(t, "Range based for loops only support message iteration or use of 'range'")

    def _AsyncFor(self, t):
        self.RaiseError(t, "Async for not supported")   

    def _If(self, t):
        """
        Fairly straightforward translation to if, else if, else format
        """
        self.fill("if ")
        self.dispatch(t.test)
        self.enter()
        self.dispatch(t.body)
        self.leave()
        # collapse nested ifs into equivalent elifs.
        while (t.orelse and len(t.orelse) == 1 and
               isinstance(t.orelse[0], ast.If)):
            t = t.orelse[0]
            self.fill("else if ")
            self.dispatch(t.test)
            self.enter()
            self.dispatch(t.body)
            self.leave()
        # final else
        if t.orelse:
            self.fill("else")
            self.enter()
            self.dispatch(t.orelse)
            self.leave()

    def _While(self, t):
        """
        Straightforward translation to c style while loop
        """
        self.fill("while ")
        self.dispatch(t.test)
        self.enter()
        self.dispatch(t.body)
        self.leave()
        if t.orelse:
            self.fill("else")
            self.enter()
            self.dispatch(t.orelse)
            self.leave()

    def _With(self, t):
        self.RaiseError(t, "With for not supported")

    def _AsyncWith(self, t):
        self.RaiseError(t, "Asynchronous with for not supported")

    # expr
    def _Bytes(self, t):
        self.RaiseError(t, "Bytes function not supported")

    def _Str(self, tree):
        if six.PY3:
            self.write(repr(tree.s))
        else:
            # if from __future__ import unicode_literals is in effect,
            # then we want to output string literals using a 'b' prefix
            # and unicode literals with no prefix.
            if "unicode_literals" not in self.future_imports:
                self.write(repr(tree.s))
            elif isinstance(tree.s, str):
                self.write("b" + repr(tree.s))
            elif isinstance(tree.s, unicode):
                self.write(repr(tree.s).lstrip("u"))
            else:
                assert False, "shouldn't get here"

    def _JoinedStr(self, t):
        # JoinedStr(expr* values)
        self.write("f")
        string = StringIO()
        self._fstring_JoinedStr(t, string.write)
        # Deviation from `unparse.py`: Try to find an unused quote.
        # This change is made to handle _very_ complex f-strings.
        v = string.getvalue()
        if '\n' in v or '\r' in v:
            quote_types = ["'''", '"""']
        else:
            quote_types = ["'", '"', '"""', "'''"]
        for quote_type in quote_types:
            if quote_type not in v:
                v = "{quote_type}{v}{quote_type}".format(quote_type=quote_type, v=v)
                break
        else:
            v = repr(v)
        self.write(v)

    def _FormattedValue(self, t):
        # FormattedValue(expr value, int? conversion, expr? format_spec)
        self.write("f")
        string = StringIO()
        self._fstring_JoinedStr(t, string.write)
        self.write(repr(string.getvalue()))

    def _fstring_JoinedStr(self, t, write):
        for value in t.values:
            meth = getattr(self, "_fstring_" + type(value).__name__)
            meth(value, write)

    def _fstring_Str(self, t, write):
        value = t.s.replace("{", "{{").replace("}", "}}")
        write(value)

    def _fstring_Constant(self, t, write):
        assert isinstance(t.value, str)
        value = t.value.replace("{", "{{").replace("}", "}}")
        write(value)

    def _fstring_FormattedValue(self, t, write):
        write("{")
        expr = StringIO()
        Unparser(t.value, expr)
        expr = expr.getvalue().rstrip("\n")
        if expr.startswith("{"):
            write(" ")  # Separate pair of opening brackets as "{ {"
        write(expr)
        if t.conversion != -1:
            conversion = chr(t.conversion)
            assert conversion in "sra"
            write("!{conversion}".format(conversion=conversion))
        if t.format_spec:
            write(":")
            meth = getattr(self, "_fstring_" + type(t.format_spec).__name__)
            meth(t.format_spec, write)
        write("}")

    def _Name(self, t):
        self.write(t.id)

    def _NameConstant(self, t):
        self.write(repr(t.value))

    def _Repr(self, t):
        self.write("`")
        self.dispatch(t.value)
        self.write("`")

    def _write_constant(self, value):
        if isinstance(value, (float, complex)):
            # Substitute overflowing decimal literal for AST infinities.
            self.write(repr(value).replace("inf", INFSTR))
        elif isinstance(value, str):
            self.write(f"\"{value}\"")
        else:
            self.write(repr(value))

    def _Constant(self, t):
        value = t.value
        if isinstance(value, tuple):
            self.write("(")
            if len(value) == 1:
                self._write_constant(value[0])
                self.write(",")
            else:
                interleave(lambda: self.write(", "), self._write_constant, value)
            self.write(")")
        elif value is Ellipsis: # instead of `...` for Py2 compatibility
            self.write("...")
        else:
            if t.kind == "u":
                self.write("u")
            self._write_constant(t.value)

    def _Num(self, t):
        repr_n = repr(t.n)
        if six.PY3:
            self.write(repr_n.replace("inf", INFSTR))
        else:
            # Parenthesize negative numbers, to avoid turning (-1)**2 into -1**2.
            if repr_n.startswith("-"):
                self.write("(")
            if "inf" in repr_n and repr_n.endswith("*j"):
                repr_n = repr_n.replace("*j", "j")
            # Substitute overflowing decimal literal for AST infinities.
            self.write(repr_n.replace("inf", INFSTR))
            if repr_n.startswith("-"):
                self.write(")")

    def _List(self, t):
        self.write("[")
        interleave(lambda: self.write(", "), self.dispatch, t.elts)
        self.write("]")

    def _ListComp(self, t):
        self.write("[")
        self.dispatch(t.elt)
        for gen in t.generators:
            self.dispatch(gen)
        self.write("]")

    def _GeneratorExp(self, t):
        self.write("(")
        self.dispatch(t.elt)
        for gen in t.generators:
            self.dispatch(gen)
        self.write(")")

    def _SetComp(self, t):
        self.write("{")
        self.dispatch(t.elt)
        for gen in t.generators:
            self.dispatch(gen)
        self.write("}")

    def _DictComp(self, t):
        self.write("{")
        self.dispatch(t.key)
        self.write(": ")
        self.dispatch(t.value)
        for gen in t.generators:
            self.dispatch(gen)
        self.write("}")

    def _comprehension(self, t):
        if getattr(t, 'is_async', False):
            self.write(" async for ")
        else:
            self.write(" for ")
        self.dispatch(t.target)
        self.write(" in ")
        self.dispatch(t.iter)
        for if_clause in t.ifs:
            self.write(" if ")
            self.dispatch(if_clause)

    def _IfExp(self, t):
        self.write("(")
        self.dispatch(t.body)
        self.write(" if ")
        self.dispatch(t.test)
        self.write(" else ")
        self.dispatch(t.orelse)
        self.write(")")

    def _Set(self, t):
        assert(t.elts) # should be at least one element
        self.write("{")
        interleave(lambda: self.write(", "), self.dispatch, t.elts)
        self.write("}")

    def _Dict(self, t):
        self.write("{")
        def write_key_value_pair(k, v):
            self.dispatch(k)
            self.write(": ")
            self.dispatch(v)

        def write_item(item):
            k, v = item
            if k is None:
                # for dictionary unpacking operator in dicts {**{'y': 2}}
                # see PEP 448 for details
                self.write("**")
                self.dispatch(v)
            else:
                write_key_value_pair(k, v)
        interleave(lambda: self.write(", "), write_item, zip(t.keys, t.values))
        self.write("}")

    def _Tuple(self, t):
        self.write("(")
        if len(t.elts) == 1:
            elt = t.elts[0]
            self.dispatch(elt)
            self.write(",")
        else:
            interleave(lambda: self.write(", "), self.dispatch, t.elts)
        self.write(")")

    unop = {"Invert":"~", "Not": "not", "UAdd":"+", "USub":"-"}
    def _UnaryOp(self, t):
        self.write("(")
        self.write(self.unop[t.op.__class__.__name__])
        self.write(" ")
        if six.PY2 and isinstance(t.op, ast.USub) and isinstance(t.operand, ast.Num):
            # If we're applying unary minus to a number, parenthesize the number.
            # This is necessary: -2147483648 is different from -(2147483648) on
            # a 32-bit machine (the first is an int, the second a long), and
            # -7j is different from -(7j).  (The first has real part 0.0, the second
            # has real part -0.0.)
            self.write("(")
            self.dispatch(t.operand)
            self.write(")")
        else:
            self.dispatch(t.operand)
        self.write(")")

    binop = { "Add":"+", "Sub":"-", "Mult":"*", "MatMult":"@", "Div":"/", "Mod":"%",
                    "LShift":"<<", "RShift":">>", "BitOr":"|", "BitXor":"^", "BitAnd":"&",
                    "FloorDiv":"//", "Pow": "**"}
    def _BinOp(self, t):

        op_name = t.op.__class__.__name__
        # translate pow into function call (no float version)
        if op_name == "Pow":
            self.write("pow(")
            self.dispatch(t.left)
            self.write(", ")
            self.dispatch(t.right)
            self.write(")")
        # translate floor div into function call (no float version)
        elif op_name == "FloorDiv":
            self.write("floor(")
            self.dispatch(t.left)
            self.write("/")
            self.dispatch(t.right)
            self.write(")")
        elif op_name == "MatMult":
            self.RaiseError(t, "Matrix multiplier operator not supported")
        else:
            self.write("(")
            self.dispatch(t.left)
            self.write(" " + self.binop[op_name] + " ")
            self.dispatch(t.right)
            self.write(")")

    cmpops = {"Eq":"==", "NotEq":"!=", "Lt":"<", "LtE":"<=", "Gt":">", "GtE":">=",
                        "Is":"==", "IsNot":"!=", "In":"in", "NotIn":"not in"}
    def _Compare(self, t):
        self.write("(")
        self.dispatch(t.left)
        for o, e in zip(t.ops, t.comparators):
            self.write(" " + self.cmpops[o.__class__.__name__] + " ")
            self.dispatch(e)
        self.write(")")

    boolops = {ast.And: 'and', ast.Or: 'or'}
    def _BoolOp(self, t):
        self.write("(")
        s = " %s " % self.boolops[t.op.__class__]
        interleave(lambda: self.write(s), self.dispatch, t.values)
        self.write(")")

    fgpufuncs = {"getID": "getID", "getVariableFloat": "getVariable<float>", "getVariableInt": "getVariable<int>"}
    fgpuattrs = ["ALIVE", "DEAD"]
    input_msg_funcs = {"getVariableFloat": "getVariable<float>", "getVariableInt": "getVariable<int>"}
    output_msg_funcs = {"setVariableFloat": "setVariable<float>", "setVariableInt": "setVariable<int>"}
    env_funcs = {"getPropertyFloat": "getProperty<float>", "getPropertyInt": "getProperty<int>"}
            
    def _Attribute(self,t):
        # Only a limited set of globals supported
        func_dict = None
        # constant is nested attribute
        if isinstance(t.value, ast.Attribute):
            # only nested attribute type is environment
            if not isinstance(t.value.value, ast.Name):
                self.RaiseError(t, "Unknown or unsupported nested attribute")
            if t.value.value.id == "FLAMEGPU" and t.value.attr == "environment":
                # check it is a supported ennvironment function
                if t.attr in self.env_funcs.keys(): 
                    # proceed
                    self.write("FLAMEGPU->environment.")
                    self.write(self.env_funcs[t.attr])
                else: 
                    self.RaiseError(t, f"Function '{t.attr}' does not exist in FLAMEGPU.environment object")
            else:
                self.RaiseError(t, f"Unknown or unsupported nested attribute in {t.value.value.id}")
        # FLAMEGPU singleton
        elif isinstance(t.value, ast.Name):
            if t.value.id == "FLAMEGPU":
                # check for legit FGPU function calls 
                if t.attr in self.fgpufuncs.keys():
                    # proceed
                    self.write("FLAMEGPU->")
                    self.write(self.fgpufuncs[t.attr])
                elif t.attr in self.fgpuattrs:
                    # proceed
                    self.write("FLAMEGPU::")
                    self.write(t.attr)
                else:
                    self.RaiseError(t, f"Function '{t.attr}' does not exist in FLAMEGPU object")
                
            # message input arg
            elif self._message_iterator_var:
                if t.value.id == self._message_iterator_var:
                    # check for legit FGPU function calls and translate
                    if t.attr in self.input_msg_funcs.keys():     
                        # proceed
                        self.write(f"{self._message_iterator_var}.")
                        self.write(self.input_msg_funcs[t.attr])
                    else:
                        self.RaiseError(t, f"Function '{t.attr}' does not exist in '{self._message_iterator_var}' message input iterable object")
                        
            # message output arg
            elif t.value.id == self._output_message_var:
                # check for legit FGPU function calls and translate
                if t.attr in self.output_msg_funcs.keys(): 
                    # proceed
                    self.write("FLAMEGPU->message_in.")
                    self.write(self.output_msg_funcs[t.attr])
                else:
                    self.RaiseError(t, f"Function '{t.attr}' does not exist in '{self._output_message_var}' message output object")
            
            # math functions (try them in raw function call format)
            elif t.value.id == "math":
                self.write(t.attr)
            # numpy types
            elif t.value.id == "numpy" or t.value.id == "np":
                if t.attr in self.numpytypes:
                    self.write(f"static_cast<{self.numpytypes[t.attr]}>")
                else: 
                    self.RaiseError(t, f"Unsupported numpy type {t.attr}")
            else:
                self.RaiseError(t, f"Global '{t.value.id}' identifiers not supported")
        else:
            self.RaiseError(t, "Unsupported function call syntax")

    def _Call(self, t):
        # check calls but let attributes check in their own dispatcher
        funcs = self._device_functions + self.pythonbuiltins
        if isinstance(t.func, ast.Name):
            if (t.func.id not in funcs):
                self.RaiseWarning(t, "Function call is not a defined FLAME GPU device function or a supported python built in.")
        self.dispatch(t.func)
        self.write("(")
        comma = False
        for e in t.args:
            if comma: self.write(", ")
            else: comma = True
            self.dispatch(e)
        if len(t.keywords):
            self.RaiseWarning(t, "Keyword argument not supported. Ignored.")
        if sys.version_info[:2] < (3, 5):
            if t.starargs:
                self.RaiseWarning(t, "Starargs not supported. Ignored.")
            if t.kwargs:
                self.RaiseWarning(t, "Kwargs not supported. Ignored.")
        self.write(")")

    def _Subscript(self, t):
        self.dispatch(t.value)
        self.write("[")
        self.dispatch(t.slice)
        self.write("]")

    def _Starred(self, t):
        self.write("*")
        self.dispatch(t.value)

    # slice
    def _Ellipsis(self, t):
        self.write("...")

    def _Index(self, t):
        self.dispatch(t.value)

    def _Slice(self, t):
        if t.lower:
            self.dispatch(t.lower)
        self.write(":")
        if t.upper:
            self.dispatch(t.upper)
        if t.step:
            self.write(":")
            self.dispatch(t.step)

    def _ExtSlice(self, t):
        interleave(lambda: self.write(', '), self.dispatch, t.dims)

    # argument
    def _arg(self, t):
        self.write(t.arg)
        if t.annotation:
            self.write(": ")
            self.dispatch(t.annotation)

    # others
    def _arguments(self, t):
        first = True
        # normal arguments
        all_args = getattr(t, 'posonlyargs', []) + t.args
        defaults = [None] * (len(all_args) - len(t.defaults)) + t.defaults
        for index, elements in enumerate(zip(all_args, defaults), 1):
            a, d = elements
            if first:first = False
            else: self.write(", ")
            self.dispatch(a)
            if d:
                self.write("=")
                self.dispatch(d)
            if index == len(getattr(t, 'posonlyargs', ())):
                self.write(", /")

        # varargs, or bare '*' if no varargs but keyword-only arguments present
        if t.vararg or getattr(t, "kwonlyargs", False):
            if first:first = False
            else: self.write(", ")
            self.write("*")
            if t.vararg:
                if hasattr(t.vararg, 'arg'):
                    self.write(t.vararg.arg)
                    if t.vararg.annotation:
                        self.write(": ")
                        self.dispatch(t.vararg.annotation)
                else:
                    self.write(t.vararg)
                    if getattr(t, 'varargannotation', None):
                        self.write(": ")
                        self.dispatch(t.varargannotation)

        # keyword-only arguments
        if getattr(t, "kwonlyargs", False):
            for a, d in zip(t.kwonlyargs, t.kw_defaults):
                if first:first = False
                else: self.write(", ")
                self.dispatch(a),
                if d:
                    self.write("=")
                    self.dispatch(d)

        # kwargs
        if t.kwarg:
            if first:first = False
            else: self.write(", ")
            if hasattr(t.kwarg, 'arg'):
                self.write("**"+t.kwarg.arg)
                if t.kwarg.annotation:
                    self.write(": ")
                    self.dispatch(t.kwarg.annotation)
            else:
                self.write("**"+t.kwarg)
                if getattr(t, 'kwargannotation', None):
                    self.write(": ")
                    self.dispatch(t.kwargannotation)

    def _keyword(self, t):
        if t.arg is None:
            # starting from Python 3.5 this denotes a kwargs part of the invocation
            self.write("**")
        else:
            self.write(t.arg)
            self.write("=")
        self.dispatch(t.value)

    def _Lambda(self, t):
        self.write("(")
        self.write("lambda ")
        self.dispatch(t.args)
        self.write(": ")
        self.dispatch(t.body)
        self.write(")")

    def _alias(self, t):
        self.write(t.name)
        if t.asname:
            self.write(" as "+t.asname)

    def _withitem(self, t):
        self.dispatch(t.context_expr)
        if t.optional_vars:
            self.write(" as ")
            self.dispatch(t.optional_vars)

def roundtrip(filename, output=sys.stdout):
    if six.PY3:
        with open(filename, "rb") as pyfile:
            encoding = tokenize.detect_encoding(pyfile.readline)[0]
        with open(filename, "r", encoding=encoding) as pyfile:
            source = pyfile.read()
    else:
        with open(filename, "r") as pyfile:
            source = pyfile.read()
    tree = compile(source, filename, "exec", ast.PyCF_ONLY_AST, dont_inherit=True)
    Unparser(tree, output)



def testdir(a):
    try:
        names = [n for n in os.listdir(a) if n.endswith('.py')]
    except OSError:
        print("Directory not readable: %s" % a, file=sys.stderr)
    else:
        for n in names:
            fullname = os.path.join(a, n)
            if os.path.isfile(fullname):
                output = StringIO()
                print('Testing %s' % fullname)
                try:
                    roundtrip(fullname, output)
                except Exception as e:
                    print('  Failed to compile, exception is %s' % repr(e))
            elif os.path.isdir(fullname):
                testdir(fullname)

def main(args):
    if args[0] == '--testdir':
        for a in args[1:]:
            testdir(a)
    else:
        for a in args:
            roundtrip(a)

if __name__=='__main__':
    main(sys.argv[1:])
