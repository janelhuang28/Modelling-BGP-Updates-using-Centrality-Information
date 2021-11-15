class NodePrefix:
    def __init__(self, addr):
        self.addr = addr
        self.children = {}
        self.node = None

    def add_child(self, child_prefix):
        if child_prefix not in self.children:
            self.children[child_prefix] = NodePrefix(child_prefix)
        return self.children[child_prefix]

    def set_node(self, node):
        self.node = node

    def find_child(self, addr):
        if addr not in self.children:
            return None
        return self.children[addr]

    def find_all_children(self, network, announcing_node):
        for child_string in self.children:
            child = self.children[child_string]
            child.child_find(network, announcing_node)

    def child_find(self, network, announcing_node):
        if self.node != None:
            if self.node.is_in_prefix(network):
                self.node.add_neighbour(announcing_node)
                announcing_node.add_neighbour(self.node)
            return
        else:
            for child_string in self.children:
                child = self.children[child_string]
                child.child_find(network, announcing_node)

    def get_all_children(self):
        all_children = []
        for child_prefix in self.children:
            child_node = self.children[child_prefix]
            self.get_child(child_node, all_children)
        return all_children

    def get_child(self, prefix_node, all_children):
        if prefix_node.node != None:
            all_children.append(prefix_node.node)
        else:
            for child_prefix in prefix_node.children:
                child_node = prefix_node.children[child_prefix]
                self.get_child(child_node, all_children)

    def print_children(self, indent):
        indent += "     "
        for child in self.children:
            print(indent + child)
            self.children[child].print_children(indent)
