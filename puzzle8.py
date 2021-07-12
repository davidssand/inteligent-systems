from copy import copy, deepcopy
from pprint import pprint
from typing import List
import time

def as_position_matrix(state, include_None=False):
    """Represents a matrix as a (item, (x_coordinate, y_coordinates)) list, sorted by item.
    This facilitates the calculation of the manhatten distance.

    E.g.
        >>> matrix = [
            [1, 2],
            [-1, -2],
        ]
        >>> as_position_matrix(objective_state)
        [
            (-2, (1, 1),
            (-1, (1, 0)),
            (1, (0, 0)),
            (2, (0, 1)),
        ]

    """
    return sorted([
        (item, (i, j)) for i, row in enumerate(state)
        for j, item in enumerate(row) if (include_None or item is not None)
    ])


initial_state = [
    [None, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
]

objective_state = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, None],
]

objective_state_position_matrix = as_position_matrix(objective_state)


class Puzzle8Solver():
    MAX_ITERATIONS = 10000

    def __init__(self, heuristic='manhattan'):
        self.heuristic = heuristic

    def update_decision_border_max_size(self, decision_border_max_size, decision_border):
        decision_border_size = len(decision_border)
        if decision_border_size > decision_border_max_size:
            return decision_border_size
        return decision_border_max_size

    def solve(self):
        print(f'Initial state \n{initial_state}\n')
        print(f'Objective state  \n{objective_state}\n')
        time0 = time.time()

        initial_node = Node(initial_state)

        decision_border = DecisionBorder([
            Path([initial_node], heuristic=self.heuristic)
        ])
        visited_nodes = NodesList([])

        iterations = 0
        number_of_nodes_created = 0
        decision_border_max_size = len(decision_border)

        while iterations < self.MAX_ITERATIONS:
            iterations += 1

            # Update decision border max size
            decision_border_max_size = self.update_decision_border_max_size(
                decision_border_max_size, decision_border)

            # Get cheapest path and decision node
            cheapest_path = decision_border.cheapest_path()
            decision_node = cheapest_path.decision_node

            # Remove cheapest from decision_border if decision_node was already visited
            if visited_nodes.contains(decision_node):
                decision_border.remove_cheapest_path()
                continue

            visited_nodes.append(decision_node)

            # Check for mission accomplished
            if decision_node.is_objective():
                break

            # Generate children
            children = decision_node.generate_children()
            number_of_nodes_created += len(children)

            # Remove cheapest path
            decision_border.remove_cheapest_path()

            # Add new paths to decision border
            for child in children:
                new_path = copy(cheapest_path)
                new_path.append(child)
                decision_border = add_path(decision_border, new_path)
        else:
            raise StopIteration('Max iterations reached.')

        print('Found solution!\n')

        print(f'Duration: {time.time() - time0} s\n')
        print(f'Iterations: {iterations}\n')
        print(f'Number of visited nodes: {len(visited_nodes)}\n')
        print(f'Number of created nodes: {number_of_nodes_created}\n')
        print(f'Maximum decision border size: {decision_border_max_size}\n')
        print(f'Cheapest path size: {len(cheapest_path)}\n')

        # visited_nodes_formated = '\n\n'.join(
        #     [str(node) for node in visited_nodes])
        # print(f'\nVisited nodes: \n{visited_nodes_formated}\n')
        print('\n------------------\n\n')
        print(f'Cheapest path:\n{cheapest_path}')


class Node():
    def __init__(self, state):
        self.state = state

    def __repr__(self):
        return '\n'.join([str(row) for row in self.state])

    def manhattan_distance(self, position_matrix_1, position_matrix_2):
        return sum([
            abs(i1 - i2) + abs(j1 - j2)
            for (_, (i1, j1)), (_, (i2, j2)) in zip(
                position_matrix_1, position_matrix_2
            )
        ])

    def manhattan_heuristic(self):
        """Sums manhattan distances of all items to their objective."""
        return self.manhattan_distance(
            as_position_matrix(self.state),
            objective_state_position_matrix,
        )

    def boolean_heuristic(self):
        """Sums number of pieces not in their objective position."""
        return sum([
            item != objective_state[i][j] for i, row in enumerate(self.state) for j, item in enumerate(row)
        ])

    def get_heuristic(self, heuristic):
        if heuristic == 'manhattan':
            return self.manhattan_heuristic()
        if heuristic == 'boolean':
            return self.boolean_heuristic()
        if heuristic is None:
            return 0
        raise ValueError('Heuristic not implemented.')

    def is_objective(self):
        return self.state == objective_state

    def empty_space_position(self):
        """Gets the position of the empty space."""
        return next(
            ((i, j) for i, row in enumerate(self.state)
             for j, item in enumerate(row) if item is None)
        )

    def possible_movements(self, i, j):
        """Return a list of possible movements."""
        i_up = (max(0, i-1), j)
        i_down = (min(len(self.state)-1, i+1), j)
        j_left = (i, max(0, j-1))
        j_right = (i, min(len(self.state[0])-1, j+1))

        return [pos for pos in (i_up, i_down, j_left, j_right) if pos != (i, j)]

    def generate_children(self) -> List:
        """Generate child states and return a list of nodes."""
        i, j = self.empty_space_position()
        children = []
        for new_i, new_j in self.possible_movements(i, j):
            child_state = deepcopy(self.state)
            child_state[i][j] = self.state[new_i][new_j]
            child_state[new_i][new_j] = None
            children.append(
                type(self)(child_state)
            )
        return children


class NodesList(list):
    def contains(self, node_to_find):
        return any([node_to_find.state == node.state for node in self])


class Path(NodesList):
    def __init__(self, *args, heuristic, **kwargs):
        super().__init__(*args, **kwargs)
        self.heuristic = heuristic

    def __repr__(self):
        return '\n    |\n    V\n'.join([str(node) for node in self])

    @property
    def decision_node(self):
        return self[-1]

    def cost(self):
        return len(self) + self.decision_node.get_heuristic(heuristic=self.heuristic)


class DecisionBorder(list):
    def __repr__(self):
        a = '\n============\n============\n'.join([str(path) for path in self])
        return f"[\n{a}\n]"

    def cheapest_path(self):
        return self[0]

    def remove_cheapest_path(self):
        self.pop(0)


def add_path(decision_border: DecisionBorder, new_path: Path) -> DecisionBorder:
    """Inserts Path into DecisionBorder, making DecisionBorder a list of ordered Path's by cost."""

    if len(decision_border) == 0:
        return DecisionBorder([new_path])

    for path_index, path in enumerate(decision_border):
        if path.cost() > new_path.cost():
            break

    return DecisionBorder(decision_border[:path_index] + [new_path] + decision_border[path_index:])


# Choose between "manhattan" (a*), "boolean" (a*) or None (uniform cost)
Puzzle8Solver(heuristic='manhattan').solve()
