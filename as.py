import matplotlib.pyplot as plt

def draw_decision_tree():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Node style settings
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=2)
    leaf_props_pos = dict(boxstyle="circle,pad=0.3", fc="lightgreen", ec="black", lw=2)
    leaf_props_neg = dict(boxstyle="circle,pad=0.3", fc="salmon", ec="black", lw=2)

    # Coordinates
    root_pos = (5, 9)
    left_node_pos = (2.5, 6)
    right_node_pos = (7.5, 6)
    
    # Leaves coordinates
    leaf_LL_pos = (1.5, 3) # Left branch, Left leaf
    leaf_LR_pos = (3.5, 3) # Left branch, Right leaf
    leaf_RL_pos = (6.5, 3) # Right branch, Left leaf
    leaf_RR_pos = (8.5, 3) # Right branch, Right leaf

    # Draw Nodes
    ax.text(root_pos[0], root_pos[1], "$a_1$", ha="center", va="center", size=20, bbox=bbox_props)
    ax.text(left_node_pos[0], left_node_pos[1], "$a_2$", ha="center", va="center", size=20, bbox=bbox_props)
    ax.text(right_node_pos[0], right_node_pos[1], "$a_2$", ha="center", va="center", size=20, bbox=bbox_props)

    # Draw Leaves
    # Left Branch (a1=T) logic: a2=T -> (+), a2=F -> (-)
    ax.text(leaf_LL_pos[0], leaf_LL_pos[1], "+", ha="center", va="center", size=20, bbox=leaf_props_pos)
    ax.text(leaf_LR_pos[0], leaf_LR_pos[1], "-", ha="center", va="center", size=20, bbox=leaf_props_neg)

    # Right Branch (a1=F) logic: a2=T -> (-), a2=F -> (+)
    ax.text(leaf_RL_pos[0], leaf_RL_pos[1], "-", ha="center", va="center", size=20, bbox=leaf_props_neg)
    ax.text(leaf_RR_pos[0], leaf_RR_pos[1], "+", ha="center", va="center", size=20, bbox=leaf_props_pos)

    # Draw Edges (Arrows)
    # From Root
    ax.annotate("", xy=left_node_pos, xytext=(5, 8.6), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(3.5, 7.5, "T", size=14, color="blue") # Label for T
    
    ax.annotate("", xy=right_node_pos, xytext=(5, 8.6), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(6.3, 7.5, "F", size=14, color="red") # Label for F

    # From Left Node (a2)
    ax.annotate("", xy=leaf_LL_pos, xytext=(2.5, 5.6), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(1.8, 4.5, "T", size=12, color="blue")
    
    ax.annotate("", xy=leaf_LR_pos, xytext=(2.5, 5.6), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(3.1, 4.5, "F", size=12, color="red")

    # From Right Node (a2)
    ax.annotate("", xy=leaf_RL_pos, xytext=(7.5, 5.6), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(6.8, 4.5, "T", size=12, color="blue")
    
    ax.annotate("", xy=leaf_RR_pos, xytext=(7.5, 5.6), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(8.1, 4.5, "F", size=12, color="red")

    plt.title("Decision Tree for Question II", size=16)
    plt.tight_layout()
    plt.show()

draw_decision_tree()