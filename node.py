from ipaddress import IPv4Address
from ipaddress import IPv6Address
from ipaddress import IPv4Network
from ipaddress import IPv6Network
from ipaddress import ip_network
import sys

class Node:
    def __init__(self, asn):
        self.asn = asn
        self.neighbours = {} # {ASN: Node}
        self.ips_ipv4 = {}
        self.ips_ipv6 = {}
        self.depth = sys.maxint
        self.reach_back = sys.maxint
        self.children = {}

    def add_neighbour(self, neighbour_to_add):
        """
        Adds neighbour, checks if it has it
        """
        self.neighbours[neighbour_to_add.asn] = neighbour_to_add

    def remove_neighbour(self, removing_neighbour):
        """
        Removes a neighbour
        """
        self.neighbours.pop(removing_neighbour.asn)

    def has_neighbour(self, neighbour):
        if neighbour.asn in self.neighbours:
            return True
        return False

    def add_ip(self, ip_to_add):
        """
        Adds ip, checks whether in list
        """
        try:
            ip_to_add_node = IPv4Address(unicode(ip_to_add))
            self.add_ip_ipv4(ip_to_add, ip_to_add_node)
            return ip_to_add_node
        except Exception as e:
            try:
                ip_to_add_node = IPv6Address(unicode(ip_to_add))
                self.add_ip_ipv6(ip_to_add, ip_to_add_node)
                return ip_to_add_node
            except Exception as e:
                print("Sorry format not supported")
                raise


    def add_ip_ipv4(self, ip_to_add, ip_to_add_node):
        self.ips_ipv4[ip_to_add] = ip_to_add_node

    def add_ip_ipv6(self, ip_to_add, ip_to_add_node):
        self.ips_ipv6[ip_to_add] = ip_to_add_node

    def get_network(self, prefix):
        try:
            return IPv4Network(unicode(prefix))
        except Exception as e:
            try:
                return IPv6Network(unicode(prefix))
            except Exception as e:
                print("Sorry format not supported")
                raise

    def withdraw_routes(self, prefix):
        """
        Withdraws prefix
        """
        prefix = prefix.replace('\n', '').replace(' ', '')
        network = self.get_network(prefix)

        neighbours_to_remove = []
        for neighbour_asn in self.neighbours:
            neighbour = self.neighbours[neighbour_asn]
            list = neighbour.ips_ipv6
            if type(network) == IPv4Network:
                list = neighbour.ips_ipv4
            for neighbour_ip in list:
                if list[neighbour_ip] in network:
                    neighbours_to_remove.append(neighbour_asn)
                    break
        # Removing neighbours
        for i in neighbours_to_remove:
            neighbour = self.neighbours.pop(i)
            neighbour.remove_neighbour(self)

    def is_in_prefix(self, network):
        """
        Checks if the prefixes of this node belong in the network
        """
        list = self.ips_ipv6
        if type(network) == IPv4Network:
            list = self.ips_ipv4
        for ip in list:
            if list[ip] in network:
                return True
        return False
