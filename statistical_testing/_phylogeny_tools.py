import numpy as np

def camin_sokal_parsimony(trait_array, species_labels, tree):
    species_trait_map = {sp:trait_array[n] for n, sp in enumerate(species_labels)}
    for node in tree.get_nonterminals(order='postorder'):
        childA = node[0]
        childB = node[1]
        trait_childA = species_trait_map[childA.name]
        trait_childB = species_trait_map[childB.name]
        if node == tree.root:
            trait = 1
        elif trait_childA + trait_childB > 0:
            trait = 1
        else:
            trait = 0
        species_trait_map[node.name] = trait
    return species_trait_map

def total_time_conserved(H, tree):
    counting_tree = tree.from_clade(tree.root)
    T = None
    for node in counting_tree.get_terminals():
        if node == counting_tree.root:
            T = 0
            break
        if H[node.name] == 0:
            counting_tree.prune(node)
    if T == None:
        T = counting_tree.total_branch_length()
    return T
