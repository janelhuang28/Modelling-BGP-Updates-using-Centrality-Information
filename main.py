from node import Node
from NodePrefix import NodePrefix
from ipaddress import IPv4Address
from ipaddress import IPv6Address
from ipaddress import IPv4Network
from ipaddress import IPv6Network
from networkx_graph import NetworkX_Graph
from centrality import Centrality
# from draw import Draw
from articulation_points import Graph
import sys

nodes = {}
nodes_prefixes = {}

def add_node(asn):
	""" Adds a node in the nodes list """
	if asn in nodes:
		return nodes[asn]
	else:
		nodes[asn] = Node(asn)
		return nodes[asn]

def print_nodes_and_write_to_file(time):
	""" prints the nodes and writes to a file """
	time = time.replace('\n', '')
	time = time.replace('TIME: ', '')
	time = time.replace('/', '-')
	print("\n" + time)
	output_file_name = time + "_test.txt"
	output_file = open(output_file_name, "w")
	for node in nodes:
		nodes_string = ''
		for neighbour_asn in nodes[node].neighbours:
			nodes_string = nodes_string + neighbour_asn + ' '
		nodes_string = node + ': ' + nodes_string
		print(nodes_string)
		output_file.write(nodes_string)
		output_file.write('\n')
	output_file.close()
	return output_file_name

def print_node_prefixes():
	""" Prints the node prefixes """
	print("\nPrinting Tree")
	for prefix in nodes_prefixes:
		node_prefix = nodes_prefixes[prefix]
		indent = ""
		print(prefix)
		node_prefix.print_children(indent)

def annnounce(prefix_to_add, announcing_node):
	""" Adds a neighbour to the announcing node if the prefix matches """
	prefix_to_add = prefix_to_add.replace(' ', '').replace('\n', '')
	network = announcing_node.get_network(prefix_to_add)
	network_splitted_prefix = prefix_to_add.split('/')
	network_range = network_splitted_prefix[1]
	subtrees = int(network_range)

	# Splits the network and finds the subtrees
	network_splitted = None
	if type(network) == IPv4Network:
		network_splitted = get_splitted('.', network_splitted_prefix[0])
		subtrees = int(subtrees / 8)
	else:
		network_exploded = IPv6Address(str(network_splitted_prefix[0])).exploded
		network_splitted = get_splitted(':', network_exploded)
		subtrees = int(subtrees / 8)

	# Finds the root node and it's children. Then adds all nodes that are in the network as neighbours
	root = find_nodes_to_add_prefix(subtrees, network_splitted)
	if root:
		root.find_all_children(network, announcing_node)

def find_nodes_to_add_prefix(subtrees, network_splitted):
	""" Finds the root node of a ip address. """
	child = None
	for i in range(subtrees):
		addr = network_splitted[i]
		if i == 0:
			if addr in nodes_prefixes:
				# Finding in the nodes prefixes
				child = nodes_prefixes[addr]
			else:
				return None
		elif child == None:
			return None
		else:
			# Finding in each node prefix object
			child = child.find_child(addr)
	return child

def get_splitted(splitting, prefix):
	""" Gets the splitted string of a prefix """
	return prefix.split(splitting)

def add_as_path(line):
	""" Adds an AS path """
	ases = line.split(' ')
	count = 0
	for i in range(len(ases)):
		if count == 0:
			count += 1
			continue
		# Always find the node
		as1 = ases[i].replace('\n', '')
		if "{" in as1:
			count += 1
			continue
		node = add_node(as1 +'')
		if count + 1 >= len(ases):
			continue
		# Adding it's neighbours
		as2 = ases[i + 1].replace('\n', '')
		if "{" in as2:
			count += 1
			continue
		node2 = add_node(as2 +'')
		node.add_neighbour(node2)
		node2.add_neighbour(node)
		count += 1

def add_ip_to_node(line, is_bview):
	""" Adds an ip to a node """
	split_line = line.split(' ')
	if not is_bview:
		ip = split_line[1]
		asn = split_line[2].replace('AS', '').replace('\n', '')
	else:
		ip = split_line[0].replace('FROM:', '')
		asn = split_line[1].replace('AS', '').replace('\n', '')
	node = add_node(asn)
	ip_node = node.add_ip(ip)
	# Adds into the node prefix tree
	add_prefix(ip, node, ip_node)
	return node


def add_prefix(ip, node, ip_node):
	""" Adds a prefix to a node """
	if type(ip_node) == IPv4Address:
		# IPv4 split by .
		ip_splited = ip.split('.')
		add_prefix_to_prefixes(ip_splited, 4, node)
	else:
		# IPv6, need to explode the address then add by the splitted form
		ipv6_address = str(IPv6Address(str(ip)).exploded)
		ip_splited = ipv6_address.split(':')
		add_prefix_to_prefixes(ip_splited, 8, node)

def add_prefix_to_prefixes(ip, version, node):
	""" Adds a prefix to the node prefix tree """
	node_prefix = None
	for i in range(version):
		addr = ip[i]
		if i == 0:
			# Gets the node prefix for the address in the node prefix tree
			node_prefix = get_node_prefix(addr)
		else:
			# Adds a child to each node prefix object
			node_prefix = node_prefix.add_child(addr)
			if i == version - 1:
				# Sets the node if we are on the last subtree
				node_prefix.set_node(node)

def get_node_prefix(addr):
	""" Gets a node prefix """
	if addr not in nodes_prefixes:
		nodes_prefixes[addr] = NodePrefix(addr)
	return nodes_prefixes[addr]

def read_updates(file_to_read, to_write_to_file):
	""" Reads the updates from a file """
	print("\nReading Updates from " + file_to_read)
	is_bview = False
	# if "bview" in file_to_read:
	# 	is_bview = True
	# Opening the file
	try:
		with open(file_to_read, 'r') as f:
			lines = f.readlines()
		f.close()
	except IOError as e:
		print("File not found!")
		return

	is_update = False
	announcing_node = None
	length = len(lines)
	# Read through the lines in the file
	for i in range(length):
		line = lines[i]
		# print(line)
		if line != '':
			if line[0] == 'A' and line[1] == 'S':
				# Finding the AS path and adding to the nodes
				add_as_path(line)
			elif line[:4] == 'FROM' or line[:2] == 'TO':
				node = add_ip_to_node(line, is_bview)
				if line[:4] == 'FROM':
					announcing_node = node
			elif line == 'WITHDRAW\n':
				i += 1
				prefix_to_withdraw = lines[i]
				while i < length and prefix_to_withdraw != '\n' and prefix_to_withdraw != 'ANNOUNCE\n':
					announcing_node.withdraw_routes(prefix_to_withdraw)
					i += 1
					if i < length:
						prefix_to_withdraw = lines[i]
				if prefix_to_withdraw == 'ANNOUNCE\n':
					i -= 1 # Retracing back
			elif line == 'ANNOUNCE\n':
				i += 1
				prefix_to_add = lines[i]
				while i < length and prefix_to_add != '\n':
					annnounce(prefix_to_add, announcing_node)
					i += 1
					if i < length:
						prefix_to_add = lines[i]
	if lines and to_write_to_file:
		time = lines[0]
		return print_nodes_and_write_to_file(time)


# Main Function
# read_updates('b_view_sample.txt') #grt.rib.20-08-29_23_19.txt # bview_sample.txt
# time_string = "XX"
# read_updates('grt.rib.20-08-29_23_19.txt')
# read_updates('grt.rib.20-08-30_23_19.txt')
# read_updates('Domestic Data/domestic.rib.20-08-29_23_19.txt')
# read_updates('Domestic Data/domestic.updates.20-08-29_23_19.txt')
# read_updates('RIPE/bview.2003_01_25_16_00.txt')

time_string = sys.argv[1] #"20-08-30_09_19"
entire_or_subnetwork = sys.argv[2]
normal_or_abnormal = sys.argv[3]
read_updates('bviews/grt.rib.' + time_string + '.txt', False)
time_string = read_updates('updates/grt.updates.' + time_string + '.txt', True)
# read_updates('Domestic Data/domestic.updates.20-08-24_11_19.txt')
# Create a graph given in the above diagram

# print "\nArticulation points in first graph "
# graph = Graph(nodes, nodes_prefixes)
# graph.AP()
# Draw(nodes, graph.APs)

# Find Centralities
centrality = Centrality(nodes, entire_or_subnetwork)
# centrality.remove_non_art_pts(graph.APs)
output_file_name_degree = "features_degree2_" + normal_or_abnormal + "_" + entire_or_subnetwork
output_file_name_closeness = "features_closeness2_" + normal_or_abnormal + "_" + entire_or_subnetwork
centrality.find_features(output_file_name_degree, time_string, output_file_name_closeness)


# Allows update input files
# update_file = raw_input("Please specify the input file: ")
# update_file = 'updates_test2.txt'
#read_updates(update_file)
