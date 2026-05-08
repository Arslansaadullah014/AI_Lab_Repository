import math

def minimax(depth, node_index, is_maximizing, values, max_depth):
    if depth== max_depth:
        return values[node_index]
    
    if is_maximizing:
        best = -math.inf
        for i in range(2):
            val = minimax(depth+1, node_index*2+i, False, values, max_depth)
            best= max(best, val)
        return best
    else:
        best= math.inf
        for i in range(2):
            val= minimax(depth+1, node_index*2+i, True, values, max_depth)
            best= min(best, val)
        return best
    
values= [3,5,6,9,12,5,23,2]
tree_depth = int(math.log2(len(values)))

optimal = minimax(0,0,True,values,tree_depth)
print("Leaf Values:", values)
print("Tree Depth:", tree_depth)
print("The Optimal Value is:", optimal)