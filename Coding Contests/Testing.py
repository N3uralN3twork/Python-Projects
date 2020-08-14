class Tree:
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None

bst = Tree()
bst.data = "A"
bst.left = Tree()
bst.right = Tree()
bst.right.left = Tree()
bst.left.left = Tree()
bst.left.left.right = Tree()
bst.left.data = "B"
bst.right.data = "C"
bst.right.left.data = "F"
bst.left.left.data = ["D", "E"]
bst.left.left.right.data = "F"



