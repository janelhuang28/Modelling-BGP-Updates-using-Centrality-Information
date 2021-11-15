# Python program to find articulation points in an undirected graph

import sys
from ap_tuple import AP_Tuple
#This class represents an undirected graph
#using adjacency list representation
class Graph:
	def __init__(self, nodes, nodes_prefixes):
		self.graph = nodes # default dictionary to store graph
		self.APs = {}
		self.nodes_prefixes = nodes_prefixes

	def find_all_edges(self):
		output_file = open("edges.txt", "w")
		edges = {}
		all_edges = []
		for node_asn in self.graph:
			node = self.graph[node_asn]
			for neighbour_asn in node.neighbours:
				edge = node_asn + " " + neighbour_asn
				edge_reverse = neighbour_asn + " " + node_asn
				if edge not in edges and edge_reverse not in edges:
					# edges[edge] = ""
					output_file.write(edge)
					output_file.write('\n')
		print(edges)

	def generate_prefix_matrices(self):
		output_file_name = "network_matrix.txt"
		output_file = open(output_file_name, "w")
		for first_prefix_ip in self.nodes_prefixes:
			first_prefix_node = self.nodes_prefixes[first_prefix_ip]
			for second_prefix_ip in first_prefix_node.children:
				second_prefix_node = first_prefix_node.children[second_prefix_ip]
				all_children = second_prefix_node.get_all_children()
				output_file.write(self.get_matrix(all_children))
				output_file.write('\n')
		output_file.close()

	def AP(self):
		""" Finds the articulation points of a graph """
		root_asn = None
		for node in self.graph:
			root_asn = node
			break

		root = self.graph[root_asn]
		numSubTrees = 0
		root.depth = 0
		# Recursively searching]
		print(root.asn)
		for neighbour_asn in root.neighbours:
			neighbour = root.neighbours[neighbour_asn]
			if neighbour.depth == sys.maxint:
				self.iter_art_pts(neighbour, 1, root) # recursive DFS for the neighbour
				numSubTrees += 1
		if numSubTrees > 1:
			self.APs[root_asn] = root
		print(self.APs)
		self.load_partitions()

	def iter_art_pts(self, node, depth, root):
		""" Iterative DFS search for a specific node """
		list = []
		stack_element = AP_Tuple(node, depth, root)
		list.append(stack_element)
		while len(list) != 0:
			top_element = list[-1]
			depth_top = top_element.depth
			node_top = top_element.node
			parent_top = top_element.parent
			if  node_top.depth == sys.maxint:
				node_top.depth = depth_top
				node_top.reach_back = depth_top
				children = []
				for neighbour_asn in node_top.neighbours:
					if neighbour_asn == parent_top.asn:
						continue
					neighbour = node_top.neighbours[neighbour_asn]
					children.append(neighbour)
				node_top.children = children
			elif len(node_top.children) != 0:
				child = node_top.children.pop()
				if child.depth < sys.maxint:
					node_top.reach_back = min(child.depth, node_top.reach_back)
				else:
					list.append(AP_Tuple(child, depth_top + 1, node_top))
			else:
				if node_top.asn != node.asn:
					parent_top.reach_back = min(node_top.reach_back, parent_top.reach_back)
					if node_top.reach_back >= parent_top.depth:
						self.APs[parent_top.asn] = parent_top
				list.pop()

	def load_partitions(self):
		""" Finds the partition of a graph """
		output_file_name = "art_pts.txt"
		output_file = open(output_file_name, "w")
		output_file_matrix = open("art_pts_matrix.txt", "w")
		all_nodes = {} # Don't add the node if a partition already has it
		for ap_asn in self.APs:
			ap = self.APs[ap_asn]
			nodes_string = ap_asn + ': '
			current_nodes = []
			current_nodes.append(ap)
			for neighbour_asn in ap.neighbours:
				if neighbour_asn in self.APs or neighbour_asn in all_nodes:
					continue
				neighbour = ap.neighbours[neighbour_asn]
				nodes_string = nodes_string + neighbour_asn + ' '
				all_nodes[neighbour_asn] = neighbour
				current_nodes.append(neighbour)
			print(nodes_string)
			output_file_matrix.write(self.get_matrix(current_nodes))
			output_file_matrix.write('\n')
			output_file.write(nodes_string)
			output_file.write('\n')
		output_file.close()

	def get_matrix(self, current_nodes):
		cols = len(current_nodes)
		rows = cols
		matrix = [[0 for i in range(cols)] for j in range(rows)]
		matrix_string = "\t"
		for node in current_nodes:
			matrix_string += node.asn + '\t'
		matrix_string += "\n"
		for row in range(rows):
			node = current_nodes[row]
			for col in range(cols):
				if col == 0:
					matrix_string += node.asn + '\t'
				neighbour = current_nodes[col]
				if node.has_neighbour(neighbour):
					matrix[row][col] = 1
				matrix_string += str(matrix[row][col]) + '\t'
			matrix_string += "\n"
		return matrix_string
