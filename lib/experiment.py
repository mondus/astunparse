import ast
import sys
import inspect
import astunparse
import astpretty

def func1():
    return 1
    
    
agent_func = """

@flamegpu_device_function
def helper(x: int):
    return x**2

@flamegpu_agent_function
def pred_output_location(message_in: MessageBruteForce, message_out: MessageBruteForce):
    id = FLAMEGPU.getID()
    offset = 10
    x = FLAMEGPU.getVariableFloat("x")
    e = FLAMEGPU.environment.getPropertyFloat("e")
    id = id+offset
    if id > 100:
        id += 8
    elif id is not 1:
        id = 2
    else:
        id = math.sin(id)
        
    id = helper(id)
    message_out.setVariableInt("id", id)
    message_out.setVariableFloat("x", x)
    
    for x in range(1, 7, 3):
        id += x
      
    for msg in message_in: 
        msg_x = msg.getVariableFloat("x");

    return FLAMEGPU.ALIVE
"""

# introspection to convert to raw string
#tree = ast.parse(inspect.getsource(pred_output_location))
tree = ast.parse(agent_func)
print(astunparse.dump(tree))
#astpretty.pprint(tree.body[0])

# try unparse
code = astunparse.unparse(tree)
print(code)

