import networkx as nx
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from collections import deque
from node import Node
import csv
import os.path

MAX_HOP_COUNT = 2
class Centrality:
    def __init__(self, nodes, entire_or_subnetwork):
        # Find centrality
        self.nodes = nodes
        self.graph = nx.Graph()
        self.subset_nodes = []
        self.get_all_nodes_subset()
        self.load_graph(entire_or_subnetwork)
        # self.find_subset_graph()

    def get_all_nodes_subset(self):
        """ Loads all nodes from this subset into a list so that features are passed """
        subset_nodes_file = open("New Zealand Nodes2 (hopcount=2).txt", "r")
        lines = subset_nodes_file.readlines()
        subset_nodes_file.close()
        for line in lines:
            nodes = line.split("\t")
            for node_asn in nodes:
                if node_asn != "" and node_asn != "\n":
                    if "\n" in node_asn:
                        node_asn = node_asn.replace("\n", "")
                    self.subset_nodes.append(node_asn)

    def remove_non_art_pts(self, art_pts):
        """ Used in the subset graph, where only the articulation points are considered """
        subset_nodes_file = open("New Zealand Nodes (hopcount=2).txt", "r")
        subset_nodes_file2 = open("New Zealand Nodes2 (hopcount=2).txt", "w")
        lines = subset_nodes_file.readlines()
        subset_nodes_file.close()
        for line in lines:
            self.subset_nodes = []
            nodes = line.split("\t")
            for node_asn in nodes:
                if node_asn != "" and node_asn != "\n":
                    if node_asn in art_pts:
                        self.subset_nodes.append(node_asn)
            for node_asn in self.subset_nodes:
                subset_nodes_file2.write(node_asn + "\t")
            subset_nodes_file2.write("\n")
            print(len(self.subset_nodes))

    def find_subset_graph(self):
        """ Gets the X hop away from nz """
        # Get nz node
        node_nz = self.nodes["38022"]
        print("174")
        print(len(self.nodes["174"].neighbours))
        all_queue_items = {}
        queue = deque()
        queue.append(node_nz)
        output_file = open("New Zealand Nodes (hopcount=" + str(MAX_HOP_COUNT) + ").txt", "w")
        output_file.write(node_nz.asn + "\t")
        level = 1 # Hop count
        queue.append(None)
        all_queue_items[node_nz.asn] = node_nz
        current_level_nodes = []
        print(len(self.graph.nodes))
        # loop till queue is empty
        while queue:
            # dequeue front node and print it
            node_current = queue.popleft()
            if node_current != None:
                current_level_nodes.append(node_current)
            if node_current == None:
                level += 1
                queue.append(None)
                output_file.write("\n")
                print(len(current_level_nodes))
                current_level_nodes = []
                # self.graph =
                if queue[0] == None or level > MAX_HOP_COUNT:
                    break # You are encountering two consecutive `nulls` means, you visited all the nodes.
                else:
                    continue
            else:
                # Add neighbors in
                for neighbour_asn in node_current.neighbours:
                    if neighbour_asn not in all_queue_items and neighbour_asn != "397878" and neighbour_asn != "328121" and neighbour_asn != "9068" and neighbour_asn != "38539":
                        neighbour = self.nodes[neighbour_asn]
                        all_queue_items[neighbour_asn] = neighbour
                        queue.append(neighbour)
                        output_file.write(neighbour_asn + "\t")


    def load_graph(self, entire_or_subnetwork):
        """ Loads the graph according to the subset. If a node is no longer connected to items in the graph, then we add it as an empty node. """
        dict_nodes = self.nodes
        if "subnetwork" in entire_or_subnetwork:
            dict_nodes = self.subset_nodes
        for node_asn in dict_nodes: # Change to nodes if want entire graph
            node = self.nodes[node_asn]
            for neighbour_asn in node.neighbours:
                neighbour = node.neighbours[neighbour_asn]
                self.graph.add_edge(node_asn, neighbour_asn)
            if len(node.neighbours) == 0:
                self.graph.add_node(node_asn)

        print(len(self.graph.nodes))

    def set_between(self, path, in_between_nodes):
        """
        Counts the number of times a node is between a paths
        """
        path_length = len(path)
        if path_length == 2:
            return
        for i in range(path_length):
            if i == 0 or i == path_length - 1:
                continue
            asn = path[i]
            self.nodes[asn].add_in_between()
            if asn not in in_between_nodes:
                in_between_nodes[asn] = 0

    def find_features(self, output_file_name_degree, time_string, output_file_closeness):
        """
        Count the number of shortest paths between nodes other than the selected node
        Count the number of shortest paths that cross over with the node in the path
        """
        # print("Degree: ")
        # print(nx.degree_centrality(self.graph))
        # print("Betweeness: ")
        # print(nx.betweenness_centrality(self.graph))
        # print("Closeness: ")
        # print(nx.closeness_centrality(self.graph))
        # Loads all the paths for the nodes and finds the betweeness centrality
        print("\nPrinting")
        number_of_nodes = len(self.graph.nodes)
        # Write to degree file
        # if not os.path.exists(output_file_name_degree):
        #     output_file_degree = open(output_file_name_degree+".csv", "w")
        #     output_file_degree.write("Time/AS" + "\t")
        #     for node_asn in self.subset_nodes:
        #         output_file_degree.write(node_asn + "\t")
        # else:
        output_file_degree = open(output_file_name_degree+".csv", "a")
        output_file_degree.write('\nt=' + time_string + "\t")

        # Write to closeness file
        # if not os.path.exists(output_file_closeness):
        #     output_file_closeness = open(output_file_closeness+".csv", "w")
        #     output_file_closeness.write("Time/AS" + "\t")
        #     for node_asn in self.subset_nodes:
        #         output_file_closeness.write(node_asn + "\t")
        # else:
        output_file_closeness = open(output_file_closeness+".csv", "a")
        output_file_closeness.write('\nt=' + time_string + "\t")

        # Finding centralitiies
        for node_asn in self.subset_nodes:
            if node_asn.startswith("0"):
                node = Node(node_asn)
            elif node_asn not in self.nodes:
                output_file_closeness.write("-\t")
                output_file_degree.write("-\t")
                continue
            else:
                node = self.nodes[node_asn]
            # Finding closeness centrality
            paths = nx.single_source_shortest_path_length(self.graph, node_asn)
            total_length = 0
            for path in paths:
                total_length += paths[path]
            node.set_path_lengths(paths)
            node.total_shortest_path_length = total_length
            node.find_closeness_centrality(number_of_nodes)
            # Finding degree centrality
            node.set_degree(number_of_nodes)

            # Printing
            print("degree: " + str(node.degree))
            print(node_asn)
            print("Closeness: " + str(node.closeness_centrality))
            output_file_closeness.write(str(node.closeness_centrality) + "\t")
            output_file_degree.write(str(node.degree) + "\t")
        output_file_degree.close()
        output_file_closeness.close()
