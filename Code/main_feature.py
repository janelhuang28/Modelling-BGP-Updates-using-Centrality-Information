from articulation_points import Graph
from node import Node
from centrality import Centrality
import sys
nodes = {}
def load_nodes(filename):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        f.close()
    except IOError as e:
        print("File not found!")

    for line in lines:
        ases = line.split(' ')
        current_node_asn = ases[0].replace(':', '')
        remaining_ases_length = len(ases)
        node = add_node(current_node_asn)
        for asn_index in range(remaining_ases_length):
            if asn_index == 0 or asn_index == remaining_ases_length - 1:
                continue
            asn_neighbour = ases[asn_index]
            neighbour_node = add_node(asn_neighbour)
            node.add_neighbour(neighbour_node)
        # print(line)

def add_node(asn):
	""" Adds a node in the nodes list """
	if asn in nodes:
		return nodes[asn]
	else:
		nodes[asn] = Node(asn)
		return nodes[asn]

time_string_file_name = sys.argv[1] #"20-08-30_09_19"
entire_or_subnetwork = sys.argv[2]
normal_or_abnormal = sys.argv[3]
time_string_file_name_stripped = sys.argv[4]
load_nodes(str(time_string_file_name)) # Data
# print(nodes)

# Find Articulation points
# print "\nArticulation points in first graph "
# graph = Graph(nodes, {})
# graph.AP()
# print(graph.APs)

# # Find Centralities
centrality = Centrality(nodes, entire_or_subnetwork)
# centrality.remove_non_art_pts(graph.APs)
output_file_name = "features_degree_" + normal_or_abnormal + "_" + entire_or_subnetwork
centrality.find_features(output_file_name, time_string_file_name_stripped)
