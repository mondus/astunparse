import ast
import sys
import inspect
import codegen
import astpretty

def func1():
    return 1
    
    
agent_func = """
@flamegpu_device_function
def helper(x: numpy.int16) -> int :
    return x**2

@flamegpu_agent_function
def pred_output_location(message_in: MessageBruteForce, message_out: MessageBruteForce):
    id = FLAMEGPU.getID()
    offset = 10
    x = FLAMEGPU.getVariableFloatArray6("x", 2)
    y = FLAMEGPU.getVariableFloat("y")
    e = FLAMEGPU.environment.getPropertyFloat("e")
    
    rand = FLAMEGPU.random.uniformFloat()
    id = id+offset
    if id > 100+3:
        id += 8
    elif id is not 1:
        id = numpy.int16(2)
    else:
        id = math.sin(id)
    
    x = 10 if id is 1 else 5
    
    id = helper(id)
    message_out.setVariableInt("id", id)
    message_out.setVariableFloat("x", x)
    
    for x in range(1+6):
        id += x
      
    for msg in message_in: 
        msg_x = msg.getVariableFloat("x");

    return FLAMEGPU.ALIVE
"""


# introspection to convert to raw string
#tree = ast.parse(inspect.getsource(pred_output_location))
tree = ast.parse(agent_func)
#print(codegen.dump(tree))
#astpretty.pprint(tree.body[0])

# try unparse
code = codegen.codegen(tree)
print(code)

