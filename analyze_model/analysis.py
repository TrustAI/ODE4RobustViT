import os 
import pandas as pd 
from tqdm import tqdm
from pathlib import Path
import torch.nn as nn 
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader

from analyze_model.util import *    

class DictInNode:
    def __init__(self, dict_node: dict={}):
        self.dict_node = dict_node
        self.children = {} 
        self.parent = {}

    def __len__(self):
        return len(self.dict_node)

    # def add_items(self, keys:list, values:list): # keys and values can be any iterable 
    #     assert len(keys) == len(values), 'DictInNode error: keys and values have to be paired' 
    #     self.dict_node.update(dict(zip(keys, values)))

class ModuleTree(DictInNode, nn.Module):
    def __init__(self, dict_node):
        super(ModuleTree, self).__init__(dict_node)
        
        assert 'named_module' in self.dict_node, 'nnTree error: the node in module tree should at least contain the module and its name'
        assert hasattr(self.dict_node['named_module'][1], 'children'), 'nnTree error: the module has to have method .children'

        try:
            next(self.dict_node['named_module'][1].children())
        except StopIteration:
            self.depth = 0 # depth is 0 for tree with no children 
            return None 

        depth_subtree = [] # to store the depth of sub trees 
        sibling_index = 0
        for sub_module in self.dict_node['named_module'][1].children():
            sub_name = f'{sub_module.__class__.__name__}({sibling_index})'
            sub_dict_node = {}

            sub_dict_node['named_module'] = (sub_name, sub_module)
            for key in set(dict_node).difference({'named_module'}):
                self.dict_node[key].assign_child_params()
                sub_dict_node[key] = self.dict_node[key].add_child(name=sub_name)

            # add and initialize the children node, cannot define as nnTree() and add info in dict_node later,
            # since information in dict_node has to be initialized at first place 
            self.children[sub_name] = ModuleTree(sub_dict_node)  
            # recursively assign parent for each node 
            self.children[sub_name].parent[self.dict_node['named_module'][0]] = self # add parent

            depth_subtree.append(self.children[sub_name].depth) # append the depth of the sub trees                 
            sibling_index += 1 

        self.depth = 1 + max(depth_subtree) # the depth of parent tree is 1 + that of the child tree with maximum depth  

        # initialized when self.decompose is called
        self.decomposed_named_modules = None 

    def __str__(self) -> str:
        
        return self.dict_node['named_module'][1].__str__()

    def __repr__(self) -> str:
        return self.dict_node['named_module'][0]

    def is_leaf(self):
        if self.children == {}: 
            is_leaf = True
        else: 
            is_leaf = False
        return is_leaf

    def is_root(self):
        if self.parent == {}: 
            is_parent = True
        else: 
            is_parent = False
        return is_parent

    def get_subtrees_of_level(self, level):

        if level == 0 or self.children == {}: # for leaf and level 0 return module itself 
            return [{self.dict_node['named_module'][0]: ModuleTree({'named_module': (self.dict_node['named_module'][0], self.dict_node['named_module'][1])})}]              

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
            self.decomposed_named_modules.append((tree_value.dict_node['named_module'][0], tree_value.dict_node['named_module'][1]))
    


class Analysis():
    def __init__(self, named_dataset, named_network, device):
        
        self.dataset_name, self.dataset = named_dataset
        self.net_name, self.net = named_network
        self.device = device

        # initialized when analysis is called  
        self.module_tree = None 
 
    def save_to_log(self, result, to_approx):
        log = pd.DataFrame(result)
        if to_approx:
            suffix = 'estimation'
        else:
            suffix = 'exact'

        folder_path = f'./analyze_model/log/{self.dataset_name}/' 
        file_name = f'{self.net_name}_{suffix}.csv' 

        if not os.path.exists(folder_path + file_name):
            Path(folder_path).mkdir(parents=True, exist_ok=True)                        
            log.to_csv(folder_path + file_name, mode='a', header=True)
        else: 
            log.to_csv(folder_path + file_name, mode='a', header=False)


    def analyze(self, frac_size, decompose_route, to_approx = False):

        # take a subset of the dataset to analyze 
        _, sub_dataset = random_split(self.dataset, [len(self.dataset) - frac_size, frac_size])
        dataloader = DataLoader(sub_dataset,
                                batch_size=1, # each batch contains only one image 
                                num_workers=4,
                                pin_memory=True,
                                shuffle=False)         

        # decompose model as basic modules 
        self.module_tree = ModuleTree({'named_module': (self.net_name, self.net)})
        self.module_tree.decompose(decompose_route)
        
        for img, _ in tqdm(dataloader):
            result = {}
            img = img.to(self.device)
            feature = self.module_tree.decomposed_named_modules[0][1](img) # output of patch embedding
     
            for named_module in self.module_tree.decomposed_named_modules[1:-1]:
                if to_approx:
                    max_sv = max_sv_estimate(named_module[1], feature)
                else: 
                    max_sv = max_sv_compute(named_module[1], feature)                                             
                result[named_module[0]] = [max_sv.item()]
                
                feature = named_module[1](feature) # the output of module_1 is the input of module_2

            self.save_to_log(result, to_approx)
             




