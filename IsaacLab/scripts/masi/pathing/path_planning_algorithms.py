"""
Path Planning Algorithms - NumPy Implementation
================================================

Unified path planning algorithms for robotics applications.
All algorithms use NumPy arrays only (no PyTorch dependencies).

Includes:
- A* (A-star) pathfinding with 3D/2D support
- RRT (Rapidly-exploring Random Tree)
- RRT* (RRT with rewiring optimization)
- PRM (Probabilistic Roadmap)

All algorithms:
- Accept tuning parameters from configuration
- Use shared ObstacleChecker for consistency
- Return NumPy arrays (float32, shape [N, 3])
- Support both 3D and planar (X-Z) planning

Refactored for NumPy-only usage
Date: 2025
"""

import heapq
import logging
import math
import numpy as np
import random
import time
from typing import List, Tuple, Optional, Dict, Any, Set, TypeAlias
from dataclasses import dataclass

# Import utilities from path_utils
from .path_utils import (
    GridConfig,
    ObstacleChecker,
    calculate_workspace_bounds,
    quaternion_conjugate,
    quaternion_multiply,
    rotation_matrix_to_quaternion,
    normalize_vector,
    basis_start_goal_plane,
)

# ============================================================================
# Type Aliases
# ============================================================================

Point3D: TypeAlias = Tuple[float, float, float]
PathType: TypeAlias = List[Point3D]
ObstacleData: TypeAlias = List[Dict[str, Any]]

# ============================================================================
# Logging Configuration
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# Custom Exceptions
# ============================================================================

class PathPlanningError(Exception):
    """Base exception for all path planning errors."""
    pass

class NoPathFoundError(PathPlanningError):
    """Raised when no valid path exists between start and goal."""
    pass

class CollisionError(PathPlanningError):
    """Raised when start or goal position is in collision."""
    pass

class TimeoutError(PathPlanningError):
    """Raised when planning exceeds maximum iterations."""
    pass

class InvalidInputError(PathPlanningError):
    """Raised when input parameters are invalid."""
    pass

# ============================================================================
# Algorithm Configuration Constants
# ============================================================================

# A* Constants
ASTAR_DEFAULT_MAX_ITERATIONS = 200000
ASTAR_PROGRESS_LOG_INTERVAL = 5000
ASTAR_EDGE_SAMPLES_MIN = 5
ASTAR_EDGE_SAMPLE_RESOLUTION = 0.5

# RRT Constants
RRT_DEFAULT_GOAL_SAMPLE_OFFSET = 0.05
RRT_PROGRESS_LOG_INTERVAL = 500

# PRM Constants
PRM_MAX_SAMPLE_ATTEMPTS = 1000
PRM_PROGRESS_LOG_INTERVAL_SAMPLES = 200
PRM_PROGRESS_LOG_INTERVAL_CONNECTIONS = 100


# ============================================================================
# Parameter Dataclasses (algorithm-specific + standardizer)
# ============================================================================

@dataclass
class AStarParams:
    """Parameters for A* path planning."""
    max_iterations: int = 200000


# Kept for backwards compatibility - use AStarParams instead
AStarPlanarParams = AStarParams


@dataclass
class RRTParams:
    max_iterations: int = 10000
    max_acceptable_cost: Optional[float] = None
    max_step_size: float = 0.05
    goal_bias: float = 0.1
    goal_tolerance: float = 0.02


@dataclass
class RRTStarParams(RRTParams):
    rewire_radius: float = 0.08
    minimum_iterations: int = 1000
    cost_improvement_patience: int = 5000


@dataclass
class PRMParams:
    num_samples: int = 1500
    connection_radius: float = 0.20
    max_connections_per_node: int = 15


# ============================================================================
# A* (A-Star) Pathfinding Algorithm
# ============================================================================

class AStar3D:
    """3D A* pathfinding algorithm with obstacle avoidance."""

    def __init__(self, grid_config: GridConfig, obstacle_checker: ObstacleChecker,
                 use_3d: bool = True, verbose: bool = False):
        """
        Initialize 3D A* planner.

        Args:
            grid_config: Grid configuration
            obstacle_checker: Obstacle collision checker
            use_3d: If True, use full 3D planning. If False, plan in X-Z plane only.
            verbose: If True, log detailed progress information
        """
        self.grid_config = grid_config
        self.obstacle_checker = obstacle_checker
        self.use_3d = use_3d
        self.verbose = verbose

        # Cache for grid conversions
        self._world_to_grid_cache = {}
        self._grid_to_world_cache = {}

    def world_to_grid(self, world_pos: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Convert world coordinates to grid indices."""
        cache_key = world_pos
        if cache_key in self._world_to_grid_cache:
            return self._world_to_grid_cache[cache_key]

        x = int(round((world_pos[0] - self.grid_config.bounds_min[0]) / self.grid_config.resolution))
        y = int(round((world_pos[1] - self.grid_config.bounds_min[1]) / self.grid_config.resolution))
        z = int(round((world_pos[2] - self.grid_config.bounds_min[2]) / self.grid_config.resolution))

        result = (x, y, z)
        self._world_to_grid_cache[cache_key] = result
        return result

    def grid_to_world(self, grid_pos: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert grid indices to world coordinates."""
        cache_key = grid_pos
        if cache_key in self._grid_to_world_cache:
            return self._grid_to_world_cache[cache_key]

        x = grid_pos[0] * self.grid_config.resolution + self.grid_config.bounds_min[0]
        y = grid_pos[1] * self.grid_config.resolution + self.grid_config.bounds_min[1]
        z = grid_pos[2] * self.grid_config.resolution + self.grid_config.bounds_min[2]

        result = (x, y, z)
        self._grid_to_world_cache[cache_key] = result
        return result

    def heuristic(self, a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
        """Euclidean distance heuristic."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1]) if self.use_3d else 0
        dz = abs(a[2] - b[2])
        return math.sqrt(dx*dx + dy*dy + dz*dz) * self.grid_config.resolution

    def get_neighbors(self, node: Tuple[int, int, int]) -> List[Tuple[Tuple[int, int, int], float]]:
        """Get valid neighbors of a node with their costs."""
        neighbors = []
        x, y, z = node
        world_curr = self.grid_to_world(node)

        if self.use_3d:
            # 26-connectivity in 3D
            directions = [
                # Face neighbors (6)
                (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                # Edge neighbors (12)
                (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
                (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
                (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
                # Corner neighbors (8)
                (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)
            ]
        else:
            # 8-connectivity in X-Z plane (Y fixed)
            directions = [
                (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1),  # 4-connectivity
                (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1)  # Diagonal
            ]

        for dx, dy, dz in directions:
            new_x, new_y, new_z = x + dx, y + dy, z + dz

            # Check bounds
            max_x = int((self.grid_config.bounds_max[0] - self.grid_config.bounds_min[0]) / self.grid_config.resolution)
            max_y = int((self.grid_config.bounds_max[1] - self.grid_config.bounds_min[1]) / self.grid_config.resolution)
            max_z = int((self.grid_config.bounds_max[2] - self.grid_config.bounds_min[2]) / self.grid_config.resolution)

            if not (0 <= new_x <= max_x and 0 <= new_y <= max_y and 0 <= new_z <= max_z):
                continue

            neighbor = (new_x, new_y, new_z)
            world_pos = self.grid_to_world(neighbor)

            # Corner-cutting prevention for 2D (planar) mode
            if not self.use_3d and dx != 0 and dz != 0:
                # both orthogonal steps must be free
                n1 = (new_x, y, z)           # step in X only
                n2 = (x, y, new_z)           # step in Z only
                w1 = self.grid_to_world(n1)
                w2 = self.grid_to_world(n2)
                if (not self.obstacle_checker.is_point_collision_free(w1) or
                    not self.obstacle_checker.is_point_collision_free(w2)):
                    continue

            # Check collision
            if not self.obstacle_checker.is_point_collision_free(world_pos):
                continue

            # --- Edge collision sampling to prevent skipping through thin geometry ---
            seg_len = math.sqrt(dx*dx + dy*dy + dz*dz) * self.grid_config.resolution
            # sample ~every ASTAR_EDGE_SAMPLE_RESOLUTION * resolution along the short edge
            num_samples = max(ASTAR_EDGE_SAMPLES_MIN,
                            int(seg_len / (self.grid_config.resolution * ASTAR_EDGE_SAMPLE_RESOLUTION)))
            if not self.obstacle_checker.is_line_collision_free(world_curr, world_pos, num_samples=num_samples):
                continue

            # Calculate cost (Euclidean distance)
            cost = math.sqrt(dx*dx + dy*dy + dz*dz) * self.grid_config.resolution
            neighbors.append((neighbor, cost))

        return neighbors

    def plan_path(self, start: Point3D, goal: Point3D,
                  max_iterations: int = ASTAR_DEFAULT_MAX_ITERATIONS) -> PathType:
        """
        Plan a path from start to goal using A*.

        Args:
            start: Start position in world coordinates
            goal: Goal position in world coordinates
            max_iterations: Maximum search iterations

        Returns:
            List of waypoints in world coordinates

        Raises:
            CollisionError: If start or goal is in collision
            TimeoutError: If max iterations exceeded
            NoPathFoundError: If no valid path exists
        """
        # Convert to grid coordinates
        start_grid = self.world_to_grid(start)
        goal_grid = self.world_to_grid(goal)

        # Check if start and goal are valid
        if not self.obstacle_checker.is_point_collision_free(start):
            raise CollisionError(f"Start position {start} is in collision")

        if not self.obstacle_checker.is_point_collision_free(goal):
            raise CollisionError(f"Goal position {goal} is in collision")

        # A* algorithm
        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(start_grid, goal_grid), 0, start_grid))

        came_from = {}
        cost_so_far = {start_grid: 0}

        iteration = 0
        logger.info(f"A* starting search from {start} to {goal}")
        logger.debug(f"Grid start: {start_grid}, Grid goal: {goal_grid}")
        logger.debug(f"Max iterations: {max_iterations}, 3D mode: {self.use_3d}")

        while open_set and iteration < max_iterations:
            iteration += 1

            if self.verbose and iteration % ASTAR_PROGRESS_LOG_INTERVAL == 0:
                logger.info(f"A* iteration {iteration}, open set size: {len(open_set)}")

            _, cost, current = heapq.heappop(open_set)

            # Check if we reached the goal
            if current == goal_grid:
                logger.info(f"A* path found after {iteration} iterations")
                break

            # Explore neighbors
            for neighbor, move_cost in self.get_neighbors(current):
                new_cost = cost + move_cost

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (priority, new_cost, neighbor))
                    came_from[neighbor] = current
        else:
            if iteration >= max_iterations:
                raise TimeoutError(f"A* exceeded max iterations ({max_iterations})")
            else:
                raise NoPathFoundError(f"A* found no path after {iteration} iterations")

        # Reconstruct path
        path_grid = []
        current = goal_grid
        while current is not None:
            path_grid.append(current)
            current = came_from.get(current)
        path_grid.reverse()

        # Convert to world coordinates
        path_world = [self.grid_to_world(grid_pos) for grid_pos in path_grid]

        logger.info(f"A* path reconstructed with {len(path_world)} waypoints")
        return path_world


# ============================================================================
# RRT (Rapidly-exploring Random Tree) Algorithm Family
# ============================================================================

@dataclass
class RRTNode:
    """Node for RRT tree."""
    position: Point3D
    parent: Optional['RRTNode'] = None
    children: List['RRTNode'] = None
    cost: float = 0.0

    def __post_init__(self):
        if self.children is None:
            self.children = []


class RRTBase:
    """Base class for RRT algorithms with common sampling and steering logic."""

    def __init__(self, grid_config: GridConfig, obstacle_checker: ObstacleChecker,
                 use_3d: bool = True,
                 max_step_size: float = 0.05,
                 goal_bias: float = 0.1,
                 goal_tolerance: float = 0.02,
                 verbose: bool = False):
        """
        Initialize RRT base planner.

        Args:
            grid_config: Grid configuration for bounds and resolution
            obstacle_checker: Obstacle collision checker
            use_3d: If True, use full 3D planning. If False, plan in X-Z plane only.
            max_step_size: Maximum distance for extending tree
            goal_bias: Probability of sampling toward goal (0.0-1.0)
            goal_tolerance: Distance to consider goal reached
            verbose: If True, log detailed progress information
        """
        self.grid_config = grid_config
        self.obstacle_checker = obstacle_checker
        self.use_3d = use_3d
        self.verbose = verbose

        # RRT parameters
        self.max_step_size = max_step_size
        self.goal_bias = goal_bias
        self.goal_tolerance = goal_tolerance

    def sample_random_point(self, goal: Point3D) -> Point3D:
        """Sample a random point in the configuration space."""
        # Goal biasing: occasionally sample near the goal
        if random.random() < self.goal_bias:
            # Sample in small sphere around goal
            offset_radius = RRT_DEFAULT_GOAL_SAMPLE_OFFSET
            if self.use_3d:
                offset = np.array([
                    random.uniform(-offset_radius, offset_radius),
                    random.uniform(-offset_radius, offset_radius),
                    random.uniform(-offset_radius, offset_radius)
                ])
            else:
                offset = np.array([
                    random.uniform(-offset_radius, offset_radius),
                    0.0,  # Keep Y fixed for 2D planning
                    random.uniform(-offset_radius, offset_radius)
                ])

            sampled_point = np.array(goal) + offset
            # Clamp to bounds
            sampled_point = np.clip(
                sampled_point,
                self.grid_config.bounds_min,
                self.grid_config.bounds_max
            )
            return tuple(sampled_point)

        # Regular uniform sampling
        if self.use_3d:
            return (
                random.uniform(self.grid_config.bounds_min[0], self.grid_config.bounds_max[0]),
                random.uniform(self.grid_config.bounds_min[1], self.grid_config.bounds_max[1]),
                random.uniform(self.grid_config.bounds_min[2], self.grid_config.bounds_max[2])
            )
        else:
            # 2D planning in X-Z plane
            y_fixed = (self.grid_config.bounds_min[1] + self.grid_config.bounds_max[1]) / 2
            return (
                random.uniform(self.grid_config.bounds_min[0], self.grid_config.bounds_max[0]),
                y_fixed,
                random.uniform(self.grid_config.bounds_min[2], self.grid_config.bounds_max[2])
            )

    def find_nearest_node(self, tree: List[RRTNode], point: Tuple[float, float, float]) -> RRTNode:
        """Find the nearest node in the tree to the given point."""
        min_distance = float('inf')
        nearest_node = None

        point_array = np.array(point)
        for node in tree:
            node_array = np.array(node.position)
            distance = np.linalg.norm(point_array - node_array)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node

        return nearest_node

    def steer(self, from_pos: Tuple[float, float, float],
              to_pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Steer from one position toward another, limited by max step size."""
        from_array = np.array(from_pos)
        to_array = np.array(to_pos)

        direction = to_array - from_array
        distance = np.linalg.norm(direction)

        if distance <= self.max_step_size:
            return to_pos

        # Limit to max step size
        unit_direction = direction / distance
        new_position = from_array + unit_direction * self.max_step_size

        return tuple(new_position)

    def find_near_nodes(self, tree: List[RRTNode], point: Tuple[float, float, float],
                       radius: float) -> List[RRTNode]:
        """Find all nodes within a given radius of the point."""
        near_nodes = []
        point_array = np.array(point)

        for node in tree:
            node_array = np.array(node.position)
            if np.linalg.norm(point_array - node_array) <= radius:
                near_nodes.append(node)

        return near_nodes

    def calculate_distance(self, pos1: Tuple[float, float, float],
                          pos2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two positions."""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def rewire_tree(self, tree: List[RRTNode], new_node: RRTNode, near_nodes: List[RRTNode]):
        """Rewire the tree to optimize paths through the new node."""
        for near_node in near_nodes:
            if near_node == new_node.parent:
                continue

            # Calculate potential new cost through new_node
            potential_cost = new_node.cost + self.calculate_distance(new_node.position, near_node.position)

            # If this path is better and collision-free, rewire
            if (potential_cost < near_node.cost and
                self.obstacle_checker.is_line_collision_free(new_node.position, near_node.position)):

                # Remove old parent connection
                if near_node.parent:
                    near_node.parent.children.remove(near_node)

                # Establish new parent connection
                near_node.parent = new_node
                new_node.children.append(near_node)

                # Update cost and propagate to descendants
                old_cost = near_node.cost
                near_node.cost = potential_cost
                self._propagate_cost_update(near_node, near_node.cost - old_cost)

    def _propagate_cost_update(self, node: RRTNode, cost_delta: float):
        """Recursively update costs of all descendants."""
        for child in node.children:
            child.cost += cost_delta
            self._propagate_cost_update(child, cost_delta)

    def choose_parent(self, near_nodes: List[RRTNode],
                     new_position: Tuple[float, float, float]) -> Tuple[RRTNode, float]:
        """Choose the best parent from near nodes to minimize cost."""
        best_parent = None
        best_cost = float('inf')

        for node in near_nodes:
            potential_cost = node.cost + self.calculate_distance(node.position, new_position)

            if (potential_cost < best_cost and
                self.obstacle_checker.is_line_collision_free(node.position, new_position)):
                best_parent = node
                best_cost = potential_cost

        return best_parent, best_cost

    def extract_path(self, goal_node: RRTNode) -> PathType:
        """Extract path from start to goal by following parent pointers."""
        path = []
        current = goal_node

        while current is not None:
            path.append(current.position)
            current = current.parent

        path.reverse()
        return path


class RRT(RRTBase):
    """Basic RRT algorithm (no rewiring optimization)."""

    def choose_parent(self, near_nodes: List[RRTNode], new_position: Point3D) -> Tuple[Optional[RRTNode], float]:
        """Just use nearest node, no cost optimization."""
        if not near_nodes:
            return None, 0.0
        return near_nodes[0], near_nodes[0].cost + self.calculate_distance(near_nodes[0].position, new_position)

    def rewire_tree(self, tree: List[RRTNode], new_node: RRTNode, near_nodes: List[RRTNode]):
        """No rewiring in basic RRT."""
        pass

    def plan_path(self, start: Point3D, goal: Point3D,
                  max_iterations: int = 10000,
                  max_acceptable_cost: Optional[float] = None) -> PathType:
        """
        Plan a path from start to goal using basic RRT.

        Args:
            start: Start position in world coordinates
            goal: Goal position in world coordinates
            max_iterations: Maximum planning iterations
            max_acceptable_cost: Unused in basic RRT (kept for API compatibility)

        Returns:
            List of waypoints in world coordinates

        Raises:
            CollisionError: If start or goal is in collision
            NoPathFoundError: If no valid path exists
        """
        # Check if start and goal are valid
        if not self.obstacle_checker.is_point_collision_free(start):
            raise CollisionError(f"Start position {start} is in collision")

        if not self.obstacle_checker.is_point_collision_free(goal):
            raise CollisionError(f"Goal position {goal} is in collision")

        # Calculate straight-line distance for reference
        straight_line_distance = self.calculate_distance(start, goal)

        # Initialize tree with start node
        start_node = RRTNode(position=start, cost=0.0)
        tree = [start_node]
        goal_node = None

        logger.info(f"RRT starting planning from {start} to {goal}")
        logger.debug(f"Straight-line distance: {straight_line_distance:.3f}m")
        logger.debug(f"Max iterations: {max_iterations}, 3D mode: {self.use_3d}")

        for iteration in range(max_iterations):
            if self.verbose and iteration % RRT_PROGRESS_LOG_INTERVAL == 0 and iteration > 0:
                logger.info(f"RRT iteration {iteration}, tree size: {len(tree)}")

            # Sample random point
            rand_point = self.sample_random_point(goal)

            # Find nearest node
            nearest_node = self.find_nearest_node(tree, rand_point)

            # Steer toward random point
            new_position = self.steer(nearest_node.position, rand_point)

            # Check if new position is collision-free
            if not self.obstacle_checker.is_point_collision_free(new_position):
                continue

            # Check if path to new position is collision-free
            if not self.obstacle_checker.is_line_collision_free(nearest_node.position, new_position):
                continue

            # Create new node (basic RRT: just use nearest as parent)
            new_cost = nearest_node.cost + self.calculate_distance(nearest_node.position, new_position)
            new_node = RRTNode(position=new_position, parent=nearest_node, cost=new_cost)
            nearest_node.children.append(new_node)
            tree.append(new_node)

            # Check if we reached the goal
            if self.calculate_distance(new_position, goal) <= self.goal_tolerance:
                goal_node = new_node
                logger.info(f"RRT goal reached at iteration {iteration}, cost: {goal_node.cost:.3f}m")
                break

        if goal_node is None:
            raise NoPathFoundError(f"RRT found no path after {max_iterations} iterations")

        # Extract and return path
        path = self.extract_path(goal_node)
        logger.info(f"RRT path found with {len(path)} waypoints, total cost: {goal_node.cost:.3f}m")

        return path


class RRTStar(RRTBase):
    """RRT* with rewiring optimization for improved path quality."""

    def __init__(self, grid_config: GridConfig, obstacle_checker: ObstacleChecker,
                 use_3d: bool = True,
                 max_step_size: float = 0.05,
                 goal_bias: float = 0.1,
                 goal_tolerance: float = 0.02,
                 rewire_radius: float = 0.08,
                 minimum_iterations: int = 1000,
                 cost_improvement_patience: int = 5000,
                 verbose: bool = False):
        """
        Initialize RRT* planner with rewiring optimization.

        Args:
            grid_config: Grid configuration for bounds and resolution
            obstacle_checker: Obstacle collision checker
            use_3d: If True, use full 3D planning. If False, plan in X-Z plane only.
            max_step_size: Maximum distance for extending tree
            goal_bias: Probability of sampling toward goal (0.0-1.0)
            goal_tolerance: Distance to consider goal reached
            rewire_radius: Radius for rewiring optimization
            minimum_iterations: Minimum iterations before early termination
            cost_improvement_patience: Iterations to wait for cost improvement
            verbose: If True, log detailed progress information
        """
        super().__init__(grid_config, obstacle_checker, use_3d, max_step_size,
                        goal_bias, goal_tolerance, verbose)

        # RRT* specific parameters
        self.rewire_radius = rewire_radius
        self.minimum_iterations = minimum_iterations
        self.cost_improvement_patience = cost_improvement_patience
        self.max_acceptable_cost = None  # Set during plan_path call

    def should_terminate_early(self, goal_node: Optional[RRTNode], iteration: int,
                              last_improvement_iteration: int) -> bool:
        """Check if we should terminate early based on cost threshold."""
        if iteration < self.minimum_iterations:
            return False

        if goal_node is None:
            return False

        if (self.max_acceptable_cost is not None and
            goal_node.cost <= self.max_acceptable_cost):
            logger.info(f"RRT* early termination: cost {goal_node.cost:.3f}m <= {self.max_acceptable_cost:.3f}m")
            return True

        if iteration - last_improvement_iteration > self.cost_improvement_patience:
            logger.info(f"RRT* early termination: no improvement for {self.cost_improvement_patience} iterations")
            return True

        return False

    def choose_parent(self, near_nodes: List[RRTNode], new_position: Point3D) -> Tuple[Optional[RRTNode], float]:
        """Choose the best parent from near nodes to minimize cost."""
        best_parent = None
        best_cost = float('inf')

        for node in near_nodes:
            potential_cost = node.cost + self.calculate_distance(node.position, new_position)

            if (potential_cost < best_cost and
                self.obstacle_checker.is_line_collision_free(node.position, new_position)):
                best_parent = node
                best_cost = potential_cost

        return best_parent, best_cost

    def rewire_tree(self, tree: List[RRTNode], new_node: RRTNode, near_nodes: List[RRTNode]):
        """Rewire the tree to optimize paths through the new node."""
        for near_node in near_nodes:
            if near_node == new_node.parent:
                continue

            # Calculate potential new cost through new_node
            potential_cost = new_node.cost + self.calculate_distance(new_node.position, near_node.position)

            # If this path is better and collision-free, rewire
            if (potential_cost < near_node.cost and
                self.obstacle_checker.is_line_collision_free(new_node.position, near_node.position)):

                # Remove old parent connection
                if near_node.parent:
                    near_node.parent.children.remove(near_node)

                # Establish new parent connection
                near_node.parent = new_node
                new_node.children.append(near_node)

                # Update cost and propagate to descendants
                old_cost = near_node.cost
                near_node.cost = potential_cost
                self._propagate_cost_update(near_node, near_node.cost - old_cost)

    def _propagate_cost_update(self, node: RRTNode, cost_delta: float):
        """Recursively update costs of all descendants."""
        for child in node.children:
            child.cost += cost_delta
            self._propagate_cost_update(child, cost_delta)

    def plan_path(self, start: Point3D, goal: Point3D,
                  max_iterations: int = 10000,
                  max_acceptable_cost: Optional[float] = None) -> PathType:
        """
        Plan a path from start to goal using RRT*.

        Args:
            start: Start position in world coordinates
            goal: Goal position in world coordinates
            max_iterations: Maximum planning iterations
            max_acceptable_cost: Early termination cost threshold (meters)

        Returns:
            List of waypoints in world coordinates

        Raises:
            CollisionError: If start or goal is in collision
            NoPathFoundError: If no valid path exists
        """
        # Check if start and goal are valid
        if not self.obstacle_checker.is_point_collision_free(start):
            raise CollisionError(f"Start position {start} is in collision")

        if not self.obstacle_checker.is_point_collision_free(goal):
            raise CollisionError(f"Goal position {goal} is in collision")

        # Set cost threshold
        self.max_acceptable_cost = max_acceptable_cost

        # Calculate straight-line distance for reference
        straight_line_distance = self.calculate_distance(start, goal)

        # Initialize tree with start node
        start_node = RRTNode(position=start, cost=0.0)
        tree = [start_node]
        goal_node = None

        last_improvement_iteration = 0

        logger.info(f"RRT* starting planning from {start} to {goal}")
        logger.debug(f"Straight-line distance: {straight_line_distance:.3f}m")
        logger.debug(f"Max acceptable cost: {max_acceptable_cost}m" if max_acceptable_cost else "No cost limit set")
        logger.debug(f"Max iterations: {max_iterations}, 3D mode: {self.use_3d}")
        logger.debug(f"Parameters: step={self.max_step_size}, bias={self.goal_bias}, rewire={self.rewire_radius}")

        for iteration in range(max_iterations):
            if self.verbose and iteration % RRT_PROGRESS_LOG_INTERVAL == 0 and iteration > 0:
                logger.info(f"RRT* iteration {iteration}, tree size: {len(tree)}")

            # Sample random point
            rand_point = self.sample_random_point(goal)

            # Find nearest node
            nearest_node = self.find_nearest_node(tree, rand_point)

            # Steer toward random point
            new_position = self.steer(nearest_node.position, rand_point)

            # Check if new position is collision-free
            if not self.obstacle_checker.is_point_collision_free(new_position):
                continue

            # Check if path to new position is collision-free
            if not self.obstacle_checker.is_line_collision_free(nearest_node.position, new_position):
                continue

            # Find near nodes for optimization
            near_nodes = self.find_near_nodes(tree, new_position, self.rewire_radius)

            # Choose best parent (RRT* optimization)
            best_parent, best_cost = self.choose_parent(near_nodes, new_position)

            if best_parent is None:
                # Fallback to nearest node
                best_parent = nearest_node
                best_cost = nearest_node.cost + self.calculate_distance(nearest_node.position, new_position)

            # Create new node
            new_node = RRTNode(position=new_position, parent=best_parent, cost=best_cost)
            best_parent.children.append(new_node)
            tree.append(new_node)

            # Rewire tree (RRT* optimization)
            self.rewire_tree(tree, new_node, near_nodes)

            # Check if we reached the goal
            if self.calculate_distance(new_position, goal) <= self.goal_tolerance:
                if goal_node is None or new_node.cost < goal_node.cost:
                    old_cost = goal_node.cost if goal_node else float('inf')
                    goal_node = new_node
                    last_improvement_iteration = iteration
                    logger.info(f"RRT* goal reached at iteration {iteration}, cost: {goal_node.cost:.3f}m (improved from {old_cost:.3f}m)")

            # Check for early termination
            if self.should_terminate_early(goal_node, iteration, last_improvement_iteration):
                logger.info(f"RRT* early termination at iteration {iteration}")
                break

        if goal_node is None:
            raise NoPathFoundError(f"RRT* found no path after {max_iterations} iterations")

        # Extract and return path
        path = self.extract_path(goal_node)
        logger.info(f"RRT* path found with {len(path)} waypoints, total cost: {goal_node.cost:.3f}m")

        return path


# ============================================================================
# PRM (Probabilistic Roadmap) Algorithm
# ============================================================================

class PRMNode:
    """Node in the PRM roadmap graph."""

    def __init__(self, node_id: int, position: Tuple[float, float, float]):
        self.id = node_id
        self.position = position
        self.neighbors: Set[int] = set()
        self.distances: Dict[int, float] = {}  # Distance to each neighbor

    def add_neighbor(self, neighbor_id: int, distance: float):
        """Add a bidirectional connection to another node."""
        self.neighbors.add(neighbor_id)
        self.distances[neighbor_id] = distance


class PRMRoadmap:
    """Probabilistic Roadmap graph structure."""

    def __init__(self):
        self.nodes: Dict[int, PRMNode] = {}
        self.next_node_id = 0

    def add_node(self, position: Tuple[float, float, float]) -> int:
        """Add a new node to the roadmap."""
        node_id = self.next_node_id
        self.nodes[node_id] = PRMNode(node_id, position)
        self.next_node_id += 1
        return node_id

    def add_edge(self, node1_id: int, node2_id: int, distance: float):
        """Add bidirectional edge between two nodes."""
        if node1_id in self.nodes and node2_id in self.nodes:
            self.nodes[node1_id].add_neighbor(node2_id, distance)
            self.nodes[node2_id].add_neighbor(node1_id, distance)

    def get_neighbors(self, node_id: int) -> List[Tuple[int, float]]:
        """Get neighbors of a node with their distances."""
        if node_id not in self.nodes:
            return []
        node = self.nodes[node_id]
        return [(neighbor_id, node.distances[neighbor_id])
                for neighbor_id in node.neighbors]

    def dijkstra(self, start_id: int, goal_id: int) -> List[int]:
        """Find shortest path between two nodes using Dijkstra's algorithm."""
        if start_id not in self.nodes or goal_id not in self.nodes:
            return []

        # Priority queue: (distance, node_id)
        pq = [(0.0, start_id)]
        distances = {start_id: 0.0}
        previous = {}
        visited = set()

        while pq:
            current_dist, current_id = heapq.heappop(pq)

            if current_id in visited:
                continue

            visited.add(current_id)

            if current_id == goal_id:
                # Reconstruct path
                path = []
                node_id = goal_id
                while node_id is not None:
                    path.append(node_id)
                    node_id = previous.get(node_id)
                path.reverse()
                return path

            # Check neighbors
            for neighbor_id, edge_distance in self.get_neighbors(current_id):
                if neighbor_id in visited:
                    continue

                new_distance = current_dist + edge_distance

                if neighbor_id not in distances or new_distance < distances[neighbor_id]:
                    distances[neighbor_id] = new_distance
                    previous[neighbor_id] = current_id
                    heapq.heappush(pq, (new_distance, neighbor_id))

        return []  # No path found


class PRM:
    """Probabilistic Roadmap pathfinding algorithm with obstacle avoidance."""

    def __init__(self, grid_config: GridConfig, obstacle_checker: ObstacleChecker,
                 use_3d: bool = True,
                 num_samples: int = 1000,
                 connection_radius: float = 0.12,
                 max_connections_per_node: int = 15,
                 verbose: bool = False):
        """
        Initialize PRM planner.

        Args:
            grid_config: Grid configuration for workspace bounds
            obstacle_checker: Obstacle collision checker
            use_3d: If True, use full 3D planning. If False, plan in X-Z plane only.
            num_samples: Number of random samples for roadmap
            connection_radius: Max distance for connecting nodes
            max_connections_per_node: Limit connections for efficiency
            verbose: If True, log detailed progress information
        """
        self.grid_config = grid_config
        self.obstacle_checker = obstacle_checker
        self.use_3d = use_3d
        self.verbose = verbose

        # PRM-specific parameters (now configurable!)
        self.num_samples = num_samples
        self.connection_radius = connection_radius
        self.max_connections_per_node = max_connections_per_node

        # Roadmap storage
        self.roadmap = PRMRoadmap()
        self.roadmap_built = False
        self.construction_time = 0.0

    def sample_random_point(self) -> Point3D:
        """Sample a random collision-free point in the workspace."""
        max_attempts = PRM_MAX_SAMPLE_ATTEMPTS

        for _ in range(max_attempts):
            if self.use_3d:
                point = (
                    random.uniform(self.grid_config.bounds_min[0], self.grid_config.bounds_max[0]),
                    random.uniform(self.grid_config.bounds_min[1], self.grid_config.bounds_max[1]),
                    random.uniform(self.grid_config.bounds_min[2], self.grid_config.bounds_max[2])
                )
            else:
                # 2D planning in X-Z plane
                y_fixed = (self.grid_config.bounds_min[1] + self.grid_config.bounds_max[1]) / 2
                point = (
                    random.uniform(self.grid_config.bounds_min[0], self.grid_config.bounds_max[0]),
                    y_fixed,
                    random.uniform(self.grid_config.bounds_min[2], self.grid_config.bounds_max[2])
                )

            if self.obstacle_checker.is_point_collision_free(point):
                return point

        # If we can't find a collision-free point, return a corner of workspace
        return self.grid_config.bounds_min

    def calculate_distance(self, pos1: Tuple[float, float, float],
                          pos2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two positions."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))

    def find_nearby_nodes(self, position: Tuple[float, float, float],
                         max_distance: float) -> List[Tuple[int, float]]:
        """Find all nodes within max_distance of the given position."""
        nearby_nodes = []

        for node_id, node in self.roadmap.nodes.items():
            distance = self.calculate_distance(position, node.position)
            if distance <= max_distance:
                nearby_nodes.append((node_id, distance))

        # Sort by distance and limit connections
        nearby_nodes.sort(key=lambda x: x[1])
        return nearby_nodes[:self.max_connections_per_node]

    def construct_roadmap(self):
        """Construct the PRM roadmap through sampling and connection."""
        if self.roadmap_built:
            return

        start_time = time.time()

        logger.info(f"PRM constructing roadmap with {self.num_samples} samples")

        # Phase 1: Sampling
        valid_samples = 0
        for i in range(self.num_samples * 3):  # Allow more attempts
            if valid_samples >= self.num_samples:
                break

            sample_point = self.sample_random_point()
            if self.obstacle_checker.is_point_collision_free(sample_point):
                self.roadmap.add_node(sample_point)
                valid_samples += 1

                if self.verbose and valid_samples % PRM_PROGRESS_LOG_INTERVAL_SAMPLES == 0:
                    logger.info(f"PRM sampled {valid_samples}/{self.num_samples} collision-free points")

        logger.info(f"PRM successfully sampled {valid_samples} collision-free points")

        # Phase 2: Connection
        connections_made = 0
        for node_id, node in self.roadmap.nodes.items():
            nearby_nodes = self.find_nearby_nodes(node.position, self.connection_radius)

            for nearby_id, distance in nearby_nodes:
                if nearby_id != node_id and nearby_id not in node.neighbors:
                    # Check if connection is collision-free
                    if self.obstacle_checker.is_line_collision_free(node.position,
                                                                  self.roadmap.nodes[nearby_id].position):
                        self.roadmap.add_edge(node_id, nearby_id, distance)
                        connections_made += 1

            if self.verbose and node_id % PRM_PROGRESS_LOG_INTERVAL_CONNECTIONS == 0:
                logger.info(f"PRM connected {node_id}/{len(self.roadmap.nodes)} nodes")

        self.construction_time = time.time() - start_time
        self.roadmap_built = True

        logger.info(f"PRM roadmap construction complete")
        logger.info(f"{len(self.roadmap.nodes)} nodes, {connections_made//2} edges, {self.construction_time:.2f}s")
        logger.debug(f"Average connections per node: {connections_made/len(self.roadmap.nodes):.1f}")
        logger.debug(f"Parameters: samples={self.num_samples}, radius={self.connection_radius}, 3D mode={self.use_3d}")

    def add_temporary_nodes(self, start: Tuple[float, float, float],
                           goal: Tuple[float, float, float]) -> Tuple[int, int]:
        """Add start and goal as temporary nodes to the roadmap."""
        start_id = self.roadmap.add_node(start)
        goal_id = self.roadmap.add_node(goal)

        # Connect start to nearby nodes
        nearby_to_start = self.find_nearby_nodes(start, self.connection_radius)
        for node_id, distance in nearby_to_start:
            if node_id != start_id:
                if self.obstacle_checker.is_line_collision_free(start,
                                                               self.roadmap.nodes[node_id].position):
                    self.roadmap.add_edge(start_id, node_id, distance)

        # Connect goal to nearby nodes
        nearby_to_goal = self.find_nearby_nodes(goal, self.connection_radius)
        for node_id, distance in nearby_to_goal:
            if node_id != goal_id:
                if self.obstacle_checker.is_line_collision_free(goal,
                                                               self.roadmap.nodes[node_id].position):
                    self.roadmap.add_edge(goal_id, node_id, distance)

        return start_id, goal_id

    def remove_temporary_nodes(self, start_id: int, goal_id: int):
        """Remove temporary start and goal nodes from roadmap."""
        # Remove connections first
        if start_id in self.roadmap.nodes:
            for neighbor_id in list(self.roadmap.nodes[start_id].neighbors):
                if neighbor_id in self.roadmap.nodes:
                    self.roadmap.nodes[neighbor_id].neighbors.discard(start_id)
                    if start_id in self.roadmap.nodes[neighbor_id].distances:
                        del self.roadmap.nodes[neighbor_id].distances[start_id]
            del self.roadmap.nodes[start_id]

        if goal_id in self.roadmap.nodes:
            for neighbor_id in list(self.roadmap.nodes[goal_id].neighbors):
                if neighbor_id in self.roadmap.nodes:
                    self.roadmap.nodes[neighbor_id].neighbors.discard(goal_id)
                    if goal_id in self.roadmap.nodes[neighbor_id].distances:
                        del self.roadmap.nodes[neighbor_id].distances[goal_id]
            del self.roadmap.nodes[goal_id]

    def plan_path(self, start: Point3D, goal: Point3D) -> PathType:
        """
        Plan a path from start to goal using PRM.

        Args:
            start: Start position in world coordinates
            goal: Goal position in world coordinates

        Returns:
            List of waypoints in world coordinates

        Raises:
            CollisionError: If start or goal is in collision
            NoPathFoundError: If no valid path exists in the roadmap
        """
        # Check if start and goal are valid
        if not self.obstacle_checker.is_point_collision_free(start):
            raise CollisionError(f"Start position {start} is in collision")

        if not self.obstacle_checker.is_point_collision_free(goal):
            raise CollisionError(f"Goal position {goal} is in collision")

        # Build roadmap if not already built
        if not self.roadmap_built:
            self.construct_roadmap()

        logger.info(f"PRM planning path from {start} to {goal}")

        # Add temporary start and goal nodes
        start_id, goal_id = self.add_temporary_nodes(start, goal)

        # Find shortest path in roadmap
        path_ids = self.roadmap.dijkstra(start_id, goal_id)

        if not path_ids:
            self.remove_temporary_nodes(start_id, goal_id)
            raise NoPathFoundError("PRM found no path in roadmap")

        # Convert path IDs to world coordinates
        path_world = []
        total_distance = 0.0

        for i, node_id in enumerate(path_ids):
            if node_id in self.roadmap.nodes:
                position = self.roadmap.nodes[node_id].position
                path_world.append(position)

                if i > 0:
                    prev_pos = self.roadmap.nodes[path_ids[i-1]].position
                    segment_distance = self.calculate_distance(position, prev_pos)
                    total_distance += segment_distance

        # Clean up temporary nodes
        self.remove_temporary_nodes(start_id, goal_id)

        logger.info(f"PRM path found with {len(path_world)} waypoints, length: {total_distance:.3f}m")
        logger.debug(f"Using roadmap with {len(self.roadmap.nodes)} nodes")

        return path_world


# ============================================================================
# Helper Functions
# ============================================================================

def setup_planner_environment(
    start_pos: Point3D,
    goal_pos: Point3D,
    obstacle_data: ObstacleData,
    grid_resolution: float,
    safety_margin: float
) -> Tuple[GridConfig, ObstacleChecker]:
    """
    Setup common planner environment (grid config and obstacle checker).

    Args:
        start_pos: Start position
        goal_pos: Goal position
        obstacle_data: List of obstacle dictionaries
        grid_resolution: Grid cell size in meters
        safety_margin: Additional clearance around obstacles

    Returns:
        Tuple of (grid_config, obstacle_checker)
    """
    bounds_min, bounds_max = calculate_workspace_bounds(obstacle_data, start_pos, goal_pos)

    grid_config = GridConfig(
        resolution=grid_resolution,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        safety_margin=safety_margin
    )

    obstacle_checker = ObstacleChecker(obstacle_data, safety_margin)

    return grid_config, obstacle_checker


# ============================================================================
# High-Level Wrapper Functions
# ============================================================================

def create_astar_3d_trajectory(
    start_pos: Point3D,
    goal_pos: Point3D,
    obstacle_data: ObstacleData,
    grid_resolution: float = 0.01,
    safety_margin: float = 0.02,
    max_iterations: int = ASTAR_DEFAULT_MAX_ITERATIONS,
    verbose: bool = False
) -> np.ndarray:
    """
    High-level interface to create A* trajectory with obstacle avoidance.

    Args:
        start_pos: Start position (x, y, z)
        goal_pos: Goal position (x, y, z)
        obstacle_data: List of obstacle dictionaries
        grid_resolution: Grid cell size in meters
        safety_margin: Additional clearance around obstacles
        max_iterations: Maximum A* search iterations
        verbose: If True, log detailed progress

    Returns:
        NumPy array of shape [N, 3] containing path waypoints

    Raises:
        PathPlanningError: If path planning fails
    """
    # Setup environment
    grid_config, obstacle_checker = setup_planner_environment(
        start_pos, goal_pos, obstacle_data, grid_resolution, safety_margin
    )

    # Create A* planner (always full 3D)
    planner = AStar3D(grid_config, obstacle_checker, use_3d=True, verbose=verbose)

    # Plan path (exceptions propagate up)
    path = planner.plan_path(start_pos, goal_pos, max_iterations=max_iterations)

    # Convert to numpy array
    path_array = np.array(path, dtype=np.float32)

    logger.debug(f"A* 3D created trajectory with {len(path)} waypoints")
    logger.debug(f"Workspace bounds: {grid_config.bounds_min} to {grid_config.bounds_max}")

    return path_array


def create_astar_plane_trajectory(
    start_pos: Point3D,
    goal_pos: Point3D,
    obstacle_data: ObstacleData,
    grid_resolution: float = 0.01,
    safety_margin: float = 0.02,
    max_iterations: int = ASTAR_DEFAULT_MAX_ITERATIONS,
    verbose: bool = False,
) -> np.ndarray:
    """
    Plan A* path on the vertical plane containing start→goal.

    Rotates the scene into the start→goal vertical plane so that it becomes
    the X-Z plane, plans with A* in 2D (use_3d=False), then rotates path back
    to world coordinates. This is useful for scenarios where the robot moves
    primarily in a plane defined by its start and goal positions.

    Note: This planner uses only the FIRST obstacle in obstacle_data as the
    wall to navigate around. For multi-obstacle scenarios, use the full 3D A*.

    Args:
        start_pos: Start position (x, y, z) in world coordinates
        goal_pos: Goal position (x, y, z) in world coordinates
        obstacle_data: List of obstacle dictionaries. Only the first obstacle
            is used as the wall for planar planning. Each should have:
            - "size": np.array([x, y, z]) - obstacle dimensions
            - "pos": np.array([x, y, z]) - obstacle center position
            - "rot": np.array([w, x, y, z]) - quaternion rotation (optional)
        grid_resolution: Grid cell size in meters for A* search
        safety_margin: Additional clearance around obstacles (applied by ObstacleChecker)
        max_iterations: Maximum A* search iterations
        verbose: If True, log detailed progress information

    Returns:
        NumPy array of shape [N, 3] containing path waypoints in world frame

    Raises:
        CollisionError: If start or goal is in collision
        TimeoutError: If max iterations exceeded
        NoPathFoundError: If no valid path exists on the plane
        InvalidInputError: If obstacle_data is empty
    """
    # Validate input
    if not obstacle_data:
        raise InvalidInputError("Planar A* requires at least one obstacle (wall) in obstacle_data")

    # Use first obstacle as the wall
    wall_obstacle = obstacle_data[0]

    # Convert to numpy arrays
    s_w = np.asarray(start_pos, dtype=np.float32)
    g_w = np.asarray(goal_pos, dtype=np.float32)

    # Build orthonormal basis for the start→goal plane
    # Xp = along start→goal, Yp = plane normal, Zp = in-plane vertical
    Xp, Yp, Zp = basis_start_goal_plane(s_w, g_w)
    R_wp = np.stack([Xp, Yp, Zp], axis=1)  # Rotation matrix: plane→world

    def world_to_plane(p_w: np.ndarray) -> np.ndarray:
        """Transform point from world frame to plane frame."""
        return R_wp.T @ (p_w - s_w)

    def plane_to_world(p_p: np.ndarray) -> np.ndarray:
        """Transform point from plane frame to world frame."""
        return (R_wp @ p_p) + s_w

    # Transform wall obstacle into plane frame
    size_w = np.asarray(wall_obstacle["size"], dtype=np.float32)
    pos_w = np.asarray(wall_obstacle["pos"], dtype=np.float32)
    rot_w = np.asarray(wall_obstacle.get("rot", [1, 0, 0, 0]), dtype=np.float32)

    # Use obstacle size in plane frame.
    size_p = size_w.astype(np.float32).copy()
    # Keep plane-normal thickness minimal but non-zero so 2D planning stays stable.
    size_p[1] = max(size_p[1], 1e-3)
    # Inflate only in-plane axes (X/Z). We avoid inflating the plane-normal axis
    # so the start/goal are not marked colliding solely due to margin padding.
    size_p[0] += 2.0 * safety_margin
    size_p[2] += 2.0 * safety_margin

    # Transform position to plane frame
    pos_p = world_to_plane(pos_w)

    # Transform rotation to plane frame
    # q_plane = (q_wp)^* ⊗ q_wall
    q_wp = rotation_matrix_to_quaternion(R_wp)
    q_wp_conj = quaternion_conjugate(q_wp)
    rot_p = quaternion_multiply(q_wp_conj, rot_w)

    # Build obstacle list in plane frame
    wall_plane = [{
        "size": size_p,
        "pos": pos_p.astype(np.float32),
        "rot": rot_p.astype(np.float32)
    }]

    # Transform start/goal to plane frame
    # Y (plane normal) is index 1 and must be ~0 in planning
    s_p = world_to_plane(s_w)
    s_p[1] = 0.0
    g_p = world_to_plane(g_w)
    g_p[1] = 0.0

    # Calculate bounds in plane frame (pad generously). Include obstacle extents so inflated
    # geometry fits inside the search grid.
    half_size = size_p * 0.5
    obs_min = pos_p - half_size
    obs_max = pos_p + half_size
    pts = np.stack([s_p, g_p, obs_min, obs_max], axis=0)
    pad = np.array([0.20, 0.05, 0.20], dtype=np.float32)
    bmin = np.min(pts, axis=0) - pad
    bmax = np.max(pts, axis=0) + pad

    # Create grid configuration in plane frame
    grid_cfg = GridConfig(
        resolution=float(grid_resolution),
        bounds_min=(float(bmin[0]), float(bmin[1]), float(bmin[2])),
        bounds_max=(float(bmax[0]), float(bmax[1]), float(bmax[2])),
        safety_margin=float(safety_margin),
    )

    # Create obstacle checker and planner
    # Safety margin already baked into size_p for X/Z; keep checker margin at 0.
    obs_checker = ObstacleChecker(wall_plane, safety_margin=0.0)
    planner = AStar3D(grid_cfg, obs_checker, use_3d=False, verbose=verbose)

    logger.info(f"Planar A* starting on start→goal plane")
    logger.debug(f"Start (world): {start_pos}, Goal (world): {goal_pos}")
    logger.debug(f"Start (plane): {tuple(s_p)}, Goal (plane): {tuple(g_p)}")
    logger.debug(f"Grid resolution: {grid_resolution}m, Safety margin: {safety_margin}m")

    # Plan path in plane frame
    path_plane = planner.plan_path(
        tuple(s_p.tolist()),
        tuple(g_p.tolist()),
        max_iterations=max_iterations
    )

    # Transform path back to world frame
    path_world = []
    for p in path_plane:
        pp = np.array([p[0], 0.0, p[2]], dtype=np.float32)  # Enforce Yp=0
        pw = plane_to_world(pp)
        path_world.append(pw.tolist())

    # Convert to numpy array
    path_array = np.array(path_world, dtype=np.float32)

    logger.info(f"Planar A* found path with {len(path_world)} waypoints on start→goal plane")

    return path_array


def create_rrt_star_trajectory(start_pos: Tuple[float, float, float],
                              goal_pos: Tuple[float, float, float],
                              obstacle_data: List[Dict[str, Any]],
                              grid_resolution: float = 0.01,
                              safety_margin: float = 0.02,
                              use_3d: bool = True,
                              max_iterations: int = 10000,
                              max_acceptable_cost: Optional[float] = None,
                              max_step_size: float = 0.05,
                              goal_bias: float = 0.1,
                              rewire_radius: float = 0.08,
                              goal_tolerance: float = 0.02,
                              minimum_iterations: int = 1000,
                              cost_improvement_patience: int = 5000) -> np.ndarray:
    """
    High-level interface to create RRT* trajectory with obstacle avoidance.

    Args:
        start_pos: Start position (x, y, z)
        goal_pos: Goal position (x, y, z)
        obstacle_data: List of obstacle dictionaries
        grid_resolution: Grid cell size in meters (used for bounds)
        safety_margin: Additional clearance around obstacles
        use_3d: Use full 3D planning or just X-Z plane
        max_iterations: Maximum RRT* iterations
        max_acceptable_cost: Early termination cost threshold (meters)
        max_step_size: Maximum tree extension distance
        goal_bias: Probability of sampling toward goal (0.0-1.0)
        rewire_radius: Radius for tree rewiring
        goal_tolerance: Distance threshold to reach goal
        minimum_iterations: Minimum iterations before early stop
        cost_improvement_patience: Iterations to wait for improvement

    Returns:
        NumPy array of shape [N, 3] containing path waypoints
    """
    # Calculate workspace bounds
    bounds_min, bounds_max = calculate_workspace_bounds(obstacle_data, start_pos, goal_pos)

    # Create grid configuration
    grid_config = GridConfig(
        resolution=grid_resolution,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        safety_margin=safety_margin
    )

    # Create obstacle checker
    obstacle_checker = ObstacleChecker(obstacle_data, safety_margin)

    # Create RRT* planner with all tuning parameters
    planner = RRTStar(
        grid_config, obstacle_checker, use_3d=use_3d,
        max_step_size=max_step_size,
        goal_bias=goal_bias,
        rewire_radius=rewire_radius,
        goal_tolerance=goal_tolerance,
        minimum_iterations=minimum_iterations,
        cost_improvement_patience=cost_improvement_patience
    )

    # Plan path
    path = planner.plan_path(start_pos, goal_pos,
                             max_iterations=max_iterations,
                             max_acceptable_cost=max_acceptable_cost)

    if not path:
        raise RuntimeError("RRT* failed to find a path")

    # Convert to numpy array
    path_array = np.array(path, dtype=np.float32)

    print(f"[RRT*] Created trajectory with {len(path)} waypoints")
    print(f"[RRT*] Start: {start_pos}, Goal: {goal_pos}")
    print(f"[RRT*] Safety margin: {safety_margin}m, Max iterations: {max_iterations}")
    print(f"[RRT*] Workspace bounds: {bounds_min} to {bounds_max}")

    return path_array


def create_rrt_trajectory(start_pos: Tuple[float, float, float],
                         goal_pos: Tuple[float, float, float],
                         obstacle_data: List[Dict[str, Any]],
                         grid_resolution: float = 0.01,
                         safety_margin: float = 0.02,
                         use_3d: bool = True,
                         max_iterations: int = 10000,
                         max_acceptable_cost: Optional[float] = None,
                         max_step_size: float = 0.05,
                         goal_bias: float = 0.1,
                         goal_tolerance: float = 0.02) -> np.ndarray:
    """
    High-level interface to create RRT trajectory with obstacle avoidance.

    Args:
        start_pos: Start position (x, y, z)
        goal_pos: Goal position (x, y, z)
        obstacle_data: List of obstacle dictionaries
        grid_resolution: Grid cell size in meters (used for bounds)
        safety_margin: Additional clearance around obstacles
        use_3d: Use full 3D planning or just X-Z plane
        max_iterations: Maximum RRT iterations
        max_acceptable_cost: Early termination cost threshold (meters)
        max_step_size: Maximum tree extension distance
        goal_bias: Probability of sampling toward goal (0.0-1.0)
        goal_tolerance: Distance threshold to reach goal

    Returns:
        NumPy array of shape [N, 3] containing path waypoints
    """
    # Calculate workspace bounds
    bounds_min, bounds_max = calculate_workspace_bounds(obstacle_data, start_pos, goal_pos)

    # Create grid configuration
    grid_config = GridConfig(
        resolution=grid_resolution,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        safety_margin=safety_margin
    )

    # Create obstacle checker
    obstacle_checker = ObstacleChecker(obstacle_data, safety_margin)

    # Create RRT planner (no rewiring)
    planner = RRT(
        grid_config, obstacle_checker, use_3d=use_3d,
        max_step_size=max_step_size,
        goal_bias=goal_bias,
        goal_tolerance=goal_tolerance,
        verbose=False,
    )

    # Plan path
    path = planner.plan_path(start_pos, goal_pos,
                             max_iterations=max_iterations,
                             max_acceptable_cost=max_acceptable_cost)

    if not path:
        raise RuntimeError("RRT failed to find a path")

    # Convert to numpy array
    path_array = np.array(path, dtype=np.float32)

    print(f"[RRT] Created trajectory with {len(path)} waypoints")
    print(f"[RRT] Start: {start_pos}, Goal: {goal_pos}")
    print(f"[RRT] Safety margin: {safety_margin}m, Max iterations: {max_iterations}")
    print(f"[RRT] Workspace bounds: {bounds_min} to {bounds_max}")

    return path_array


def create_prm_trajectory(start_pos: Tuple[float, float, float],
                         goal_pos: Tuple[float, float, float],
                         obstacle_data: List[Dict[str, Any]],
                         grid_resolution: float = 0.01,
                         safety_margin: float = 0.02,
                         use_3d: bool = True,
                         num_samples: int = 1500,
                         connection_radius: float = 0.20,
                         max_connections_per_node: int = 15) -> np.ndarray:
    """
    High-level interface to create PRM trajectory with obstacle avoidance.

    Args:
        start_pos: Start position (x, y, z)
        goal_pos: Goal position (x, y, z)
        obstacle_data: List of obstacle dictionaries
        grid_resolution: Grid cell size in meters (for bounds)
        safety_margin: Additional clearance around obstacles
        use_3d: Use full 3D planning or just X-Z plane
        num_samples: Number of samples for roadmap construction
        connection_radius: Maximum distance for connecting roadmap nodes
        max_connections_per_node: Limit connections per node

    Returns:
        NumPy array of shape [N, 3] containing path waypoints
    """
    # Calculate workspace bounds
    bounds_min, bounds_max = calculate_workspace_bounds(obstacle_data, start_pos, goal_pos)

    # Create grid configuration
    grid_config = GridConfig(
        resolution=grid_resolution,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        safety_margin=safety_margin
    )

    # Create obstacle checker
    obstacle_checker = ObstacleChecker(obstacle_data, safety_margin)

    # Create PRM planner with all tuning parameters
    planner = PRM(
        grid_config, obstacle_checker, use_3d=use_3d,
        num_samples=num_samples,
        connection_radius=connection_radius,
        max_connections_per_node=max_connections_per_node
    )

    # Plan path
    path = planner.plan_path(start_pos, goal_pos)

    if not path:
        raise RuntimeError("PRM failed to find a path")

    # Convert to numpy array
    path_array = np.array(path, dtype=np.float32)

    print(f"[PRM] Created trajectory with {len(path)} waypoints")
    print(f"[PRM] Start: {start_pos}, Goal: {goal_pos}")
    print(f"[PRM] Roadmap: {num_samples} samples, {connection_radius:.3f}m connection radius")
    print(f"[PRM] Safety margin: {safety_margin}m")
    print(f"[PRM] Workspace bounds: {bounds_min} to {bounds_max}")

    return path_array
