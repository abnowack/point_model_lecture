# from numba import jit # uncomment this and the function decorators to make this run much faster
import numpy as np
import random
import igraph as ig
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def create_fission_chain_graph(pf, p_nu):
    
    class Neutron_node:
        def __init__(self, id = 0, fate = None, parent = 0):
            self.id = id
            self.fate = fate # escape or fission
            self.parent = parent
    
    g = ig.Graph()

    id = 1
    
    neutron_population = [Neutron_node(id)]
    dead_neutrons = []

    while len(neutron_population) >= 1:
        new_neutrons = []

        for i in neutron_population:
            # fission
            r = np.random.random()
            if r < pf:
                i.fate = 'fission'
                for z in range(np.random.choice(len(p_nu), p=p_nu)):
                    new_neutrons.append(Neutron_node(id=id, parent=i.id))
                    id += 1
                dead_neutrons.append(i)
            # escape
            else:
                i.fate = 'escape'
                dead_neutrons.append(i)

        neutron_population = new_neutrons
    
    g.add_vertex(0, type='source')
    for i in dead_neutrons:
        if i.parent != i.id:
            g.add_vertex(i.id, type=i.fate)
    
    for i in dead_neutrons:
        if i.parent != i.id:
            g.add_edge(i.parent, i.id)

    return g.simplify()

def plot_fission_chain_graph(graph, show_node_labels=True, show_edge_labels=True, tree_layout=True, save_filename=None, return_fig=False):

    if tree_layout:
        lo = graph.layout_reingold_tilford(root=[0])
        lo.rotate(180)
    else:
        lo = graph.layout(layout='fr')

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    node_color_lookup = {
        "source": colors[0],
        "fission": colors[1],
        "escape": colors[2],
    }

    node_colors = [node_color_lookup[vs["type"]] for vs in graph.vs]

    fig, ax = plt.subplots()
    h = ig.plot(
        graph,
        layout=lo,
        target=ax,
        vertex_color=node_colors,
        vertex_size=0.3,
        edge_background="white",
    )

    patches = []
    for rxn, c in node_color_lookup.items():
        if rxn == "source":
            patches.append(mpatches.Patch(color=c, label="Source"))
        if rxn == "escape":
            patches.append(mpatches.Patch(color=c, label="Escape"))
        if rxn == "fission":
            patches.append(mpatches.Patch(color=c, label="Fission"))
    ax.legend(handles=patches)

    if return_fig:
        return fig

    # fig.tight_layout()
    # fig.subplots_adjust(top=0.95)

def create_and_plot_fission_chain(pf, p_nu):
    chain = create_fission_chain_graph(pf, p_nu)
    plot_fission_chain_graph(chain)
    
    # total neutrons
    neutrons_total = chain.ecount()
    neutrons_leaked = sum([v['type'] == 'escape' for v in chain.vs])
    
    print(f"Total  Neutrons = {neutrons_total}")
    print(f"Leaked Neutrons = {neutrons_leaked}")
    

# @jit(nopython=True)
def int_choice(cum_weights):
    r = random.random()
    for i in range(len(cum_weights)):
        if r <= cum_weights[i]:
            return i


# @jit(nopython=True)
def get_array_item(arr, index):
    if len(arr) > index:
        return arr[index]
    else:
        return arr[-1]


# @jit(nopython=True)
def chain(reaction_data_cumsum, pf, pc, return_total=False):
    neutron_population = 1

    leaked_neutrons = 0
    captured_neutrons = 0
    total_neutrons = 1

    neutron_generation = 0

    # rest of chain
    while neutron_population >= 1:
        pf_i = get_array_item(pf, neutron_generation)
        pc_i = get_array_item(pc, neutron_generation)
        for i in range(neutron_population):
            # fission
            r = np.random.random()
            # capture
            if r < pc_i:
                neutron_population -= 1
                captured_neutrons += 1
            elif (r - pc_i) < pf_i:
                fission_neutrons = int_choice(reaction_data_cumsum)
                neutron_population += fission_neutrons - 1
                total_neutrons += fission_neutrons
            # escape
            else:
                neutron_population -= 1
                leaked_neutrons += 1

        neutron_generation += 1

    return total_neutrons, leaked_neutrons


# @jit(nopython=True)
def simulate_impl(nchains, reaction_data, pf, pc=0.0, default_max=10000):

    # neutron count histograms
    result_leaked = np.zeros(default_max)
    result_total = np.zeros(default_max)

    reaction_cumsum = np.cumsum(reaction_data)

    for _ in range(nchains):
        total_neutrons, leaked_neutrons = chain(reaction_cumsum, pf, pc)

        # expand leaked and total arrays if chain is larger than array size
        if leaked_neutrons >= result_leaked.size:
            new_result = np.zeros(leaked_neutrons + 1)
            new_result[:result_leaked.size] = result_leaked
            result_leaked = new_result
            
        if total_neutrons >= result_total.size:
            new_result = np.zeros(total_neutrons + 1)
            new_result[:result_total.size] = result_total
            result_total = new_result

        # add a count for chain creating N total and leaked neutrons
        result_leaked[leaked_neutrons] += 1
        result_total[total_neutrons] += 1
    
    # shrink histograms such that no non-zero bins are at the end
    leaked_last_nonzero_index = result_leaked.nonzero()[0].max()
    total_last_nonzero_index = result_leaked.nonzero()[0].max()
    last_nonzero_index = max(leaked_last_nonzero_index, total_last_nonzero_index)
    
    new_result_leaked = np.zeros(last_nonzero_index + 1)
    new_result_leaked = result_leaked[:last_nonzero_index + 1]
    result_leaked = new_result_leaked
    
    new_result_total = np.zeros(last_nonzero_index + 1)
    new_result_total = result_total[:last_nonzero_index + 1]
    result_total = new_result_total

    return result_total, result_leaked

def simulate(nchains, reaction_data, pf, pc=0.0, default_max=10000):
    if np.ndim(pf) == 0:
        pf = np.array([pf])
    if np.ndim(pc) == 0:
        pc = np.array([pc])

    return simulate_impl(nchains, reaction_data, pf, pc, default_max)