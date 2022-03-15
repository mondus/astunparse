# coding: utf-8
from six.moves import cStringIO
from .codegen import CodeGenerator

def codegen(tree):
    v = cStringIO()
    CodeGenerator(tree, file=v)
    return v.getvalue()

