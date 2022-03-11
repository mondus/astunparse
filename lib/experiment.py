import ast
import sys
import inspect
import astunparse
import astpretty

def func1():
    return 1
    
    
agent_func = """
def pred_output_location(message_in: MessageNone, message_out: MessageBruteForce):
    id = FLAMEGPU.getID()
    offset = 10
    x = FLAMEGPU.getVariableFloat("x")
    id = id+offset
    message_out.setVariableInt("id", id)
    message_out.setVariableFloat("x", x)

    return FLAMEGPU.ALIVE
"""

# introspection to convert to raw string
#tree = ast.parse(inspect.getsource(pred_output_location))
tree = ast.parse(agent_func)
astpretty.pprint(tree.body[0])

# try unparse
code = astunparse.unparse(tree)
print(code)

