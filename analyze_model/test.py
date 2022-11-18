class ModuleNode:
    def __init__(self, named_module):
        self.module_name, self.module = named_module 
        self.children = {}


class ModuleTree(ModuleNode):
    def __init__(self, named_module):
        super().__init__(named_module)
        # initialized when self.decompose is called
        self.decomposed_named_modules = None 
        try:
            next(self.module.children())
        except StopIteration:
            self.depth = 0 # depth is 0 for tree with no children 
            return None 

        depth_subtree = [] # to store the depth of sub trees 

        sibling_index = 0
        for sub_module in self.module.children():
            sub_name = f'{sub_module.__class__.__name__}({sibling_index})'
            self.children[sub_name] = ModuleTree((sub_name, sub_module))
            sibling_index += 1
            depth_subtree.append(self.children[sub_name].depth) # append the depth of the sub trees                 

        self.depth = 1 + max(depth_subtree) # the depth of parent tree is 1 + that of the child tree with maximum depth  


    def get_subtrees_of_level(self, level):

        if level == 0 or self.children == {}: # for leaf and level 0 return module itself 
            return [{self.module_name: ModuleTree((self.module_name, self.module))}]              

        modules = []
        current_level = level - 1
        for sub_module in self.children.values():
            modules += sub_module.get_subtrees_of_level(current_level)

        return modules 
        
    def decompose(self, decompose_route):

        # fetch the target trees 
        trees = self.get_subtrees_of_level(0) # starting from the root                         
        while len(decompose_route) > 0:
            new_trees = []
            for i, level in enumerate(decompose_route[0]):
                new_trees += next(iter(trees[i].values())).get_subtrees_of_level(level)

            trees = new_trees
            decompose_route.pop(0) # update the route until it's empty

        # change trees to modules 
        self.decomposed_named_modules = []
        for tree in trees:
            tree_value = next(iter(tree.values()))
            self.decomposed_named_modules.append((tree_value.module_name, tree_value.module))