"""
Script to generate computational graph diagrams for the requires_grad tutorial.
This creates comp-graph-1.png and comp-graph-2.png with the correct requires_grad values.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

def create_comp_graph_1():
    """Create the computational graph after forward pass (comp-graph-1.png)"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    leaf_color = '#E8F4F8'
    nonleaf_color = '#FFF4E6'
    
    # Node positions (x, y)
    nodes = {
        'x': (1, 7),
        'W': (1, 5),
        'b': (1, 3),
        'y': (1, 1),
        'z': (5, 5),
        'y_pred': (7.5, 5),
        'loss': (9, 5)
    }
    
    # Node properties: (label, is_leaf, requires_grad, retains_grad)
    node_props = {
        'x': ('x\nshape: (1, 3)', True, False, False),
        'W': ('W\nshape: (3, 2)', True, True, False),
        'b': ('b\nshape: (1, 2)', True, True, False),
        'y': ('y\nshape: (1, 2)', True, False, False),
        'z': ('z = x @ W + b\nshape: (1, 2)', False, True, False),  # CORRECTED: requires_grad=True
        'y_pred': ('y_pred = ReLU(z)\nshape: (1, 2)', False, True, False),
        'loss': ('loss = MSE(y_pred, y)\nscalar', False, True, False)
    }
    
    # Draw nodes
    for node_name, (x, y) in nodes.items():
        label, is_leaf, requires_grad, retains_grad = node_props[node_name]
        color = leaf_color if is_leaf else nonleaf_color
        
        # Draw box
        box = FancyBboxPatch((x - 0.6, y - 0.4), 1.2, 0.8,
                            boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor=color,
                            linewidth=2)
        ax.add_patch(box)
        
        # Add main label
        ax.text(x, y + 0.15, label, ha='center', va='center',
               fontsize=9, weight='bold')
        
        # Add properties
        props_text = f"is_leaf = {is_leaf}\nrequires_grad = {requires_grad}\nretains_grad = {retains_grad}"
        ax.text(x, y - 0.25, props_text, ha='center', va='top',
               fontsize=7, style='italic')
    
    # Draw arrows
    arrows = [
        ('x', 'z'),
        ('W', 'z'),
        ('b', 'z'),
        ('z', 'y_pred'),
        ('y_pred', 'loss'),
        ('y', 'loss')
    ]
    
    for start, end in arrows:
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        
        # Adjust arrow positions to start/end at box edges
        if start in ['x', 'W', 'b', 'y']:
            x1 += 0.6
        if end == 'z':
            x2 -= 0.6
        elif end == 'y_pred':
            x2 -= 0.6
        elif end == 'loss':
            x2 -= 0.6
            
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=20,
                               linewidth=2, color='black')
        ax.add_patch(arrow)
    
    # Add title
    ax.text(5, 9.5, 'Computational Graph After Forward Pass',
           ha='center', va='top', fontsize=14, weight='bold')
    
    # Add legend
    leaf_patch = mpatches.Patch(color=leaf_color, label='Leaf Tensor', edgecolor='black')
    nonleaf_patch = mpatches.Patch(color=nonleaf_color, label='Non-leaf Tensor', edgecolor='black')
    ax.legend(handles=[leaf_patch, nonleaf_patch], loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('comp-graph-1.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Generated comp-graph-1.png")
    plt.close()


def create_comp_graph_2():
    """Create the computational graph after backward pass (comp-graph-2.png)"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    leaf_color = '#E8F4F8'
    nonleaf_color = '#FFF4E6'
    
    # Node positions (x, y)
    nodes = {
        'x': (1, 7),
        'W': (1, 5),
        'b': (1, 3),
        'y': (1, 1),
        'z': (5, 5),
        'y_pred': (7.5, 5),
        'loss': (9, 5)
    }
    
    # Node properties: (label, is_leaf, requires_grad, retains_grad)
    node_props = {
        'x': ('x\nshape: (1, 3)', True, False, False),
        'W': ('W\nshape: (3, 2)', True, True, False),
        'b': ('b\nshape: (1, 2)', True, True, False),
        'y': ('y\nshape: (1, 2)', True, False, False),
        'z': ('z = x @ W + b\nshape: (1, 2)', False, True, True),  # retains_grad=True after retain_grad()
        'y_pred': ('y_pred = ReLU(z)\nshape: (1, 2)', False, True, True),  # retains_grad=True
        'loss': ('loss = MSE(y_pred, y)\nscalar', False, True, True)  # retains_grad=True
    }
    
    # Draw nodes
    for node_name, (x, y) in nodes.items():
        label, is_leaf, requires_grad, retains_grad = node_props[node_name]
        color = leaf_color if is_leaf else nonleaf_color
        
        # Draw box
        box = FancyBboxPatch((x - 0.6, y - 0.4), 1.2, 0.8,
                            boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor=color,
                            linewidth=2)
        ax.add_patch(box)
        
        # Add main label
        ax.text(x, y + 0.15, label, ha='center', va='center',
               fontsize=9, weight='bold')
        
        # Add properties
        props_text = f"is_leaf = {is_leaf}\nrequires_grad = {requires_grad}\nretains_grad = {retains_grad}"
        ax.text(x, y - 0.25, props_text, ha='center', va='top',
               fontsize=7, style='italic')
    
    # Draw arrows
    arrows = [
        ('x', 'z'),
        ('W', 'z'),
        ('b', 'z'),
        ('z', 'y_pred'),
        ('y_pred', 'loss'),
        ('y', 'loss')
    ]
    
    for start, end in arrows:
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        
        # Adjust arrow positions to start/end at box edges
        if start in ['x', 'W', 'b', 'y']:
            x1 += 0.6
        if end == 'z':
            x2 -= 0.6
        elif end == 'y_pred':
            x2 -= 0.6
        elif end == 'loss':
            x2 -= 0.6
            
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=20,
                               linewidth=2, color='black')
        ax.add_patch(arrow)
    
    # Add title
    ax.text(5, 9.5, 'Computational Graph After Backward Pass',
           ha='center', va='top', fontsize=14, weight='bold')
    
    # Add legend
    leaf_patch = mpatches.Patch(color=leaf_color, label='Leaf Tensor', edgecolor='black')
    nonleaf_patch = mpatches.Patch(color=nonleaf_color, label='Non-leaf Tensor', edgecolor='black')
    ax.legend(handles=[leaf_patch, nonleaf_patch], loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('comp-graph-2.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Generated comp-graph-2.png")
    plt.close()


if __name__ == '__main__':
    create_comp_graph_1()
    create_comp_graph_2()
    print("\nDiagrams generated successfully!")
    print("The corrected comp-graph-1.png now shows z with requires_grad=True")
