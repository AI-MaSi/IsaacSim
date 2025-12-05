#!/usr/bin/env python3
"""
Trajectory Visualizer for Excavator System

GUI-based trajectory visualization with multi-select comparison support.
Features planned vs executed trajectory overlay with obstacle rendering.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from scipy.signal import savgol_filter


class TrajectoryVisualizer:
    def __init__(self, excavator_base_offset=0.00):
        """
        Initialize the trajectory visualizer.
        
        Args:
            excavator_base_offset (float): Height offset of excavator base from ground (meters)
        """
        self.excavator_base_offset = excavator_base_offset
        self.fig = None
        self.ax = None
        
    def filter_outliers_and_smooth(self, trajectory, max_step_distance=0.03, smooth_window=71, polyorder=2):
        """
        Remove outlier points and smooth trajectory to reduce sensor noise.

        Uses physically-justified outlier removal based on robot kinematics and sampling rate,
        followed by Savitzky-Golay filtering which preserves trajectory shape while reducing noise.

        Args:
            trajectory (np.array): Nx3 array of trajectory points
            max_step_distance (float): Maximum allowed distance between consecutive points (meters).
                                       Default 0.03m (3cm) based on robot kinematics and sampling rate.
            smooth_window (int): Window size for Savitzky-Golay filter (must be odd). Default 21.
                                 Aggressive smoothing to remove encoder oscillations and high-frequency noise.
            polyorder (int): Polynomial order for Savitzky-Golay filter. Default 2.

        Returns:
            np.array: Filtered and smoothed trajectory
        """
        if len(trajectory) < 2:
            return trajectory

        # Calculate step distances between consecutive points
        step_distances = np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1))

        # Find outlier indices (points with large jumps)
        outlier_mask = np.zeros(len(trajectory), dtype=bool)
        outlier_mask[1:] = step_distances > max_step_distance

        # Remove outliers
        filtered_traj = trajectory[~outlier_mask]
        num_removed = np.sum(outlier_mask)

        if num_removed > 0:
            print(f"  Removed {num_removed} outlier points (step > {max_step_distance}m)")

        # Apply Savitzky-Golay smoothing to each axis
        # This preserves trajectory shape better than moving average
        if len(filtered_traj) >= smooth_window:
            smoothed_traj = np.zeros_like(filtered_traj)
            for i in range(3):  # x, y, z
                smoothed_traj[:, i] = savgol_filter(filtered_traj[:, i],
                                                    window_length=smooth_window,
                                                    polyorder=polyorder,
                                                    mode='nearest')
            print(f"  Applied Savitzky-Golay filter (window={smooth_window}, polyorder={polyorder})")
            return smoothed_traj
        else:
            return filtered_traj

    def load_trajectory_data(self, filepath, apply_filtering=False, max_step_distance=0.03,
                            smooth_window=71, polyorder=2):
        """
        Load trajectory data from CSV file.

        Args:
            filepath (str): Path to CSV file containing trajectory data
            apply_filtering (bool): If True, remove outliers and smooth data
            max_step_distance (float): Maximum step distance for outlier removal
            smooth_window (int): Window size for Savitzky-Golay filter
            polyorder (int): Polynomial order for Savitzky-Golay filter

        Returns:
            tuple: (planned_trajectory, executed_trajectory) as numpy arrays
        """
        try:
            df = pd.read_csv(filepath)

            # Validate required columns
            required_cols = ['x_g', 'y_g', 'z_g', 'x_e', 'y_e', 'z_e']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Extract planned trajectory (goal points)
            planned_trajectory = df[['x_g', 'y_g', 'z_g']].values

            # Extract executed trajectory
            executed_trajectory = df[['x_e', 'y_e', 'z_e']].values

            print(f"Loaded {len(planned_trajectory)} trajectory points")

            # Apply filtering if requested
            if apply_filtering:
                print("Applying filtering...")
                planned_trajectory = self.filter_outliers_and_smooth(
                    planned_trajectory, max_step_distance, smooth_window, polyorder)
                executed_trajectory = self.filter_outliers_and_smooth(
                    executed_trajectory, max_step_distance, smooth_window, polyorder)
                print(f"After filtering: {len(planned_trajectory)} points remaining")

            # Apply excavator base offset (move everything up by 0.15m)
            planned_trajectory[:, 2] += self.excavator_base_offset
            executed_trajectory[:, 2] += self.excavator_base_offset

            return planned_trajectory, executed_trajectory

        except Exception as e:
            print(f"Error loading trajectory data: {e}")
            sys.exit(1)
    
    def quaternion_to_rotation_matrix(self, quat):
        """
        Convert quaternion to rotation matrix.
        
        Args:
            quat (array-like): [w, x, y, z] quaternion
            
        Returns:
            np.array: 3x3 rotation matrix
        """
        w, x, y, z = quat
        
        # Normalize quaternion
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # Convert to rotation matrix
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        return R
    
    def create_box_vertices(self, size, pos, rot):
        """
        Create vertices for a 3D box obstacle.
        
        Args:
            size (np.array): [x, y, z] dimensions
            pos (np.array): [x, y, z] center position
            rot (np.array): [w, x, y, z] quaternion rotation
            
        Returns:
            np.array: 8x3 array of box vertices
        """
        # Create box vertices centered at origin
        half_size = size / 2
        vertices = np.array([
            [-half_size[0], -half_size[1], -half_size[2]],
            [+half_size[0], -half_size[1], -half_size[2]],
            [+half_size[0], +half_size[1], -half_size[2]],
            [-half_size[0], +half_size[1], -half_size[2]],
            [-half_size[0], -half_size[1], +half_size[2]],
            [+half_size[0], -half_size[1], +half_size[2]],
            [+half_size[0], +half_size[1], +half_size[2]],
            [-half_size[0], +half_size[1], +half_size[2]]
        ])
        
        # Apply rotation
        R = self.quaternion_to_rotation_matrix(rot)
        rotated_vertices = vertices @ R.T
        
        # Apply translation (no base offset for obstacles)
        world_vertices = rotated_vertices + pos
        
        return world_vertices
    
    def create_box_faces(self, vertices):
        """
        Create faces for a 3D box from vertices.
        
        Args:
            vertices (np.array): 8x3 array of box vertices
            
        Returns:
            list: List of face vertex arrays
        """
        # Define the 6 faces of a box (each face is a quad)
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[7], vertices[6], vertices[5]],  # top
            [vertices[0], vertices[4], vertices[5], vertices[1]],  # front
            [vertices[2], vertices[6], vertices[7], vertices[3]],  # back
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
            [vertices[1], vertices[5], vertices[6], vertices[2]]   # right
        ]
        
        return faces
    
    def plot_trajectories(self, trajectory_data_list, obstacle_data=None, title="Trajectory Visualization",
                         show_ground_plane=False, show_planned=True, show_executed=True,
                         show_obstacles=True, show_collision=True):
        """
        Create 3D visualization of multiple trajectories and obstacles.

        Args:
            trajectory_data_list (list): List of dicts with 'planned', 'executed', 'name', 'source'
            obstacle_data (list): List of obstacle dictionaries with 'size', 'pos', 'rot'
            title (str): Plot title
            show_ground_plane (bool): If True, show green reference plane at z=0
            show_planned (bool): If True, show planned trajectories
            show_executed (bool): If True, show executed trajectories
            show_obstacles (bool): If True, show physical obstacles (gray boxes)
            show_collision (bool): If True, show collision zones (red boxes)
        """
        # Create figure and 3D axis
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Color palette for multiple trajectories (planned/executed pairs)
        color_palette = [
            ('#0066CC', '#FF3333'),  # Blue/Red (default)
            ('#00AA00', '#FF8800'),  # Green/Orange
            ('#9900CC', '#FFCC00'),  # Purple/Yellow
            ('#00CCCC', '#FF0088'),  # Cyan/Magenta
            ('#CC6600', '#0088FF'),  # Brown/Light Blue
        ]

        all_points = []

        # Plot obstacles FIRST (for legend order - obstacles appear before trajectories)
        if obstacle_data:
            legend_physical = False
            legend_collision = False
            for i, obstacle in enumerate(obstacle_data):
                size_physical = obstacle.get('size')
                size_collision = obstacle.get('padded_size', size_physical)
                pos = obstacle['pos']
                rot = obstacle['rot']

                # Collision box (inflated) - translucent red (if enabled)
                if show_collision and size_collision is not None:
                    vertices = self.create_box_vertices(size_collision, pos, rot)
                    faces = self.create_box_faces(vertices)
                    collision_collection = Poly3DCollection(
                        faces,
                        alpha=0.10,
                        facecolor='red',
                        edgecolor='#aa0000',
                        zorder=1
                    )
                    self.ax.add_collection3d(collision_collection)
                    if not legend_collision:
                        self.ax.scatter([], [], [], c='red', alpha=0.3, s=100, marker='s',
                                        label='Collision box')
                        legend_collision = True

                # Physical obstacle - gray (if enabled)
                if show_obstacles and size_physical is not None:
                    vertices = self.create_box_vertices(size_physical, pos, rot)
                    faces = self.create_box_faces(vertices)
                    obstacle_collection = Poly3DCollection(
                        faces,
                        alpha=0.35,
                        facecolor='gray',
                        edgecolor='black',
                        zorder=2
                    )
                    self.ax.add_collection3d(obstacle_collection)
                    if not legend_physical:
                        self.ax.scatter([], [], [], c='gray', alpha=0.6, s=100, marker='s',
                                        label='Obstacle')
                        legend_physical = True

        # Plot Start/Goal markers (after obstacles, before trajectories for legend order)
        # Note: Different trajectories may have different start/goal positions (A→B vs B→A)
        # So we plot start/goal for EACH trajectory with its own color
        start_goal_added_to_legend = False
        if trajectory_data_list:
            for idx, traj_data in enumerate(trajectory_data_list):
                planned_traj = traj_data['planned']
                traj_color = color_palette[idx % len(color_palette)][0]  # Use planned color for markers

                # Add to legend only once
                if not start_goal_added_to_legend:
                    self.ax.scatter(planned_traj[0, 0], planned_traj[0, 1], planned_traj[0, 2],
                                   c=traj_color, s=120, marker='o', label='Start', zorder=6)
                    self.ax.scatter(planned_traj[-1, 0], planned_traj[-1, 1], planned_traj[-1, 2],
                                   c=traj_color, s=120, marker='s', label='Goal', zorder=6)
                    start_goal_added_to_legend = True
                else:
                    # Plot without label (no duplicate legend entries)
                    self.ax.scatter(planned_traj[0, 0], planned_traj[0, 1], planned_traj[0, 2],
                                   c=traj_color, s=120, marker='o', zorder=6)
                    self.ax.scatter(planned_traj[-1, 0], planned_traj[-1, 1], planned_traj[-1, 2],
                                   c=traj_color, s=120, marker='s', zorder=6)

        # Plot trajectory lines (these appear last in legend)
        for idx, traj_data in enumerate(trajectory_data_list):
            planned_traj = traj_data['planned']
            executed_traj = traj_data['executed']
            name = traj_data['name']
            source = traj_data.get('source', 'unknown')

            # Get color pair (cycle through palette)
            planned_color, executed_color = color_palette[idx % len(color_palette)]

            # Format label with source tag
            source_tag = "(sim)" if source == 'sim' else "(irl)" if source == 'real' else ""
            label_suffix = f" {source_tag}" if source_tag else ""

            # Plot planned trajectory (if enabled)
            if show_planned:
                self.ax.plot(planned_traj[:, 0], planned_traj[:, 1], planned_traj[:, 2],
                            color=planned_color, linestyle='-', linewidth=2.5,
                            label=f'{name} - Planned{label_suffix}', alpha=0.8, zorder=5)
                all_points.append(planned_traj)

            # Plot executed trajectory (if enabled)
            if show_executed:
                self.ax.plot(executed_traj[:, 0], executed_traj[:, 1], executed_traj[:, 2],
                            color=executed_color, linestyle='-', linewidth=2.5,
                            label=f'{name} - Executed{label_suffix}', alpha=0.8, zorder=5)
                all_points.append(executed_traj)

        # Add ground plane reference (conditional based on setting)
        if all_points:
            all_points_combined = np.vstack(all_points)

            if show_ground_plane:
                x_min, x_max = all_points_combined[:, 0].min() - 0.1, all_points_combined[:, 0].max() + 0.1
                y_min, y_max = all_points_combined[:, 1].min() - 0.1, all_points_combined[:, 1].max() + 0.1
                ground_level = 0.0  # Since we already offset everything by base_offset

                # Create a semi-transparent ground plane
                xx, yy = np.meshgrid([x_min, x_max], [y_min, y_max])
                zz = np.full_like(xx, ground_level)
                self.ax.plot_surface(xx, yy, zz, alpha=0.2, color='lightgreen')
        else:
            all_points_combined = np.array([[0, 0, 0]])  # fallback

        # Customize plot
        self.ax.set_xlabel('X (m)', fontsize=12)
        self.ax.set_ylabel('Y (m)', fontsize=12)
        self.ax.set_zlabel('Z (m)', fontsize=12)
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.legend(fontsize=9, loc='upper left')
        
        # Set equal aspect ratio
        max_range = np.array([
            all_points_combined[:, 0].max() - all_points_combined[:, 0].min(),
            all_points_combined[:, 1].max() - all_points_combined[:, 1].min(),
            all_points_combined[:, 2].max() - all_points_combined[:, 2].min()
        ]).max() / 2.0

        mid_x = (all_points_combined[:, 0].max() + all_points_combined[:, 0].min()) * 0.5
        mid_y = (all_points_combined[:, 1].max() + all_points_combined[:, 1].min()) * 0.5
        mid_z = (all_points_combined[:, 2].max() + all_points_combined[:, 2].min()) * 0.5
        
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Set a nice viewing angle
        self.ax.view_init(elev=20, azim=45)

        plt.tight_layout()

        # Maximize window
        manager = plt.get_current_fig_manager()
        try:
            # Try different methods depending on backend
            if hasattr(manager, 'window'):
                # TkAgg backend
                manager.window.state('zoomed')
            elif hasattr(manager, 'frame'):
                # WX backend
                manager.frame.Maximize(True)
            elif hasattr(manager, 'full_screen_toggle'):
                # Qt backend
                manager.full_screen_toggle()
        except Exception:
            # Fallback - just show normally if maximize fails
            pass

        plt.show()


def detect_algorithm_from_filename(filepath):
    """
    Algorithm name from CSV filename.

    Args:
        filepath (str): Path to CSV file

    Returns:
        str: Algorithm name
    """
    filename = Path(filepath).stem.lower()  # Get filename without extension

    # Remove _sim suffix if present for algorithm detection
    filename = filename.replace('_sim', '')

    # Check patterns (order matters - check rrtstar before rrt!)
    if filename.startswith('astar'):
        return 'A*'
    elif filename.startswith('rrtstar'):
        return 'RRT*'
    elif filename.startswith('rrt'):
        return 'RRT'
    elif filename.startswith('prm'):
        return 'PRM'
    else:
        print(f"Warning: Could not detect algorithm from filename '{filename}', using 'Unknown'")
        return 'Unknown'


class TrajectoryVisualizerGUI:
    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("Trajectory Visualizer")
        self.root.geometry("750x580")

        # Current directory for file searching
        self.current_dir = Path.cwd()

        # Obstacle data loaded from metrics
        self.obstacle_data = None

        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title with help button
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        title_label = ttk.Label(title_frame, text="Excavator Trajectory Visualizer",
                               font=("Arial", 16, "bold"))
        title_label.pack(side=tk.LEFT, padx=(0, 10))

        help_btn = ttk.Button(title_frame, text="?", width=3, command=self.show_help)
        help_btn.pack(side=tk.LEFT)

        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        # === MAIN TAB ===
        main_tab = ttk.Frame(notebook, padding="10")
        notebook.add(main_tab, text="Main")

        # Directory selection
        ttk.Label(main_tab, text="Search Directory:", font=("Arial", 10)).grid(row=0, column=0, sticky=tk.W, pady=5)

        dir_frame = ttk.Frame(main_tab)
        dir_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        self.dir_var = tk.StringVar(value=str(self.current_dir))
        dir_entry = ttk.Entry(dir_frame, textvariable=self.dir_var, width=60)
        dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        browse_btn = ttk.Button(dir_frame, text="Browse", command=self.browse_directory)
        browse_btn.pack(side=tk.LEFT)

        # CSV file selection
        ttk.Label(main_tab, text="Select Trajectory File(s) - Ctrl+Click for multi-select:", font=("Arial", 10)).grid(row=2, column=0, sticky=tk.W, pady=(10, 5))

        # Treeview with scrollbar for CSV files (supports color coding)
        list_frame = ttk.Frame(main_tab)
        list_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create Treeview with single column (enable multi-select)
        self.file_treeview = ttk.Treeview(list_frame, height=10, show='tree',
                                          selectmode='extended', yscrollcommand=scrollbar.set)
        self.file_treeview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.file_treeview.yview)

        # Configure color tags for sim (light green) and real (light red)
        self.file_treeview.tag_configure('sim', background='#d4edda')  # light green
        self.file_treeview.tag_configure('real', background='#f8d7da')  # light red
        self.file_treeview.tag_configure('unknown', background='#ffffff')  # white

        # Bind selection event to update metrics dropdown
        self.file_treeview.bind('<<TreeviewSelect>>', self.on_trajectory_select)

        # Metrics dropdown
        metrics_frame = ttk.Frame(main_tab)
        metrics_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

        ttk.Label(metrics_frame, text="Metrics:", font=("Arial", 10)).pack(side=tk.LEFT, padx=(0, 5))
        self.metrics_var = tk.StringVar(value="Select a trajectory file to view metrics")
        self.metrics_dropdown = ttk.Combobox(metrics_frame, textvariable=self.metrics_var, state='readonly', width=70)
        self.metrics_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Configure grid weights for main tab
        main_tab.columnconfigure(0, weight=1)
        main_tab.rowconfigure(3, weight=1)

        # === SETTINGS TAB ===
        settings_tab = ttk.Frame(notebook, padding="10")
        notebook.add(settings_tab, text="Settings")

        # Base offset input
        ttk.Label(settings_tab, text="Visualization Settings", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(0, 15))

        offset_frame = ttk.Frame(settings_tab)
        offset_frame.grid(row=1, column=0, sticky=tk.W, pady=5)

        ttk.Label(offset_frame, text="Base Offset (m):").pack(side=tk.LEFT, padx=(0, 5))
        self.offset_var = tk.StringVar(value="0.0")
        offset_entry = ttk.Entry(offset_frame, textvariable=self.offset_var, width=10)
        offset_entry.pack(side=tk.LEFT)
        ttk.Label(offset_frame, text="(height offset for trajectory display)", font=("Arial", 9)).pack(side=tk.LEFT, padx=(5, 0))

        # Visualization toggles section
        ttk.Label(settings_tab, text="Visualization Toggles", font=("Arial", 12, "bold")).grid(row=2, column=0, sticky=tk.W, pady=(20, 5))

        # Show planned toggle
        show_planned_frame = ttk.Frame(settings_tab)
        show_planned_frame.grid(row=3, column=0, sticky=tk.W, pady=2)

        self.show_planned_var = tk.BooleanVar(value=True)
        show_planned_checkbox = ttk.Checkbutton(show_planned_frame, text="Show Planned Trajectories",
                                                variable=self.show_planned_var)
        show_planned_checkbox.pack(side=tk.LEFT)

        # Show executed toggle
        show_executed_frame = ttk.Frame(settings_tab)
        show_executed_frame.grid(row=4, column=0, sticky=tk.W, pady=2)

        self.show_executed_var = tk.BooleanVar(value=True)
        show_executed_checkbox = ttk.Checkbutton(show_executed_frame, text="Show Executed Trajectories",
                                                 variable=self.show_executed_var)
        show_executed_checkbox.pack(side=tk.LEFT)

        # Show obstacles toggle
        show_obstacles_frame = ttk.Frame(settings_tab)
        show_obstacles_frame.grid(row=5, column=0, sticky=tk.W, pady=2)

        self.show_obstacles_var = tk.BooleanVar(value=True)
        show_obstacles_checkbox = ttk.Checkbutton(show_obstacles_frame, text="Show Obstacles",
                                                  variable=self.show_obstacles_var)
        show_obstacles_checkbox.pack(side=tk.LEFT)
        ttk.Label(show_obstacles_frame, text="(gray boxes)", font=("Arial", 9)).pack(side=tk.LEFT, padx=(5, 0))

        # Show collision zones toggle
        show_collision_frame = ttk.Frame(settings_tab)
        show_collision_frame.grid(row=6, column=0, sticky=tk.W, pady=2)

        self.show_collision_var = tk.BooleanVar(value=True)
        show_collision_checkbox = ttk.Checkbutton(show_collision_frame, text="Show Collision Zones",
                                                  variable=self.show_collision_var)
        show_collision_checkbox.pack(side=tk.LEFT)
        ttk.Label(show_collision_frame, text="(red boxes - obstacle + safety margin)", font=("Arial", 9)).pack(side=tk.LEFT, padx=(5, 0))

        # Ground plane toggle
        ground_plane_frame = ttk.Frame(settings_tab)
        ground_plane_frame.grid(row=7, column=0, sticky=tk.W, pady=2)

        self.show_ground_plane_var = tk.BooleanVar(value=False)
        ground_plane_checkbox = ttk.Checkbutton(ground_plane_frame, text="Show Ground Plane",
                                                variable=self.show_ground_plane_var)
        ground_plane_checkbox.pack(side=tk.LEFT)
        ttk.Label(ground_plane_frame, text="(green reference plane at z=0)", font=("Arial", 9)).pack(side=tk.LEFT, padx=(5, 0))

        # Filtering section
        ttk.Label(settings_tab, text="Data Filtering", font=("Arial", 12, "bold")).grid(row=8, column=0, sticky=tk.W, pady=(20, 10))

        filter_enable_frame = ttk.Frame(settings_tab)
        filter_enable_frame.grid(row=9, column=0, sticky=tk.W, pady=5)

        self.filter_var = tk.BooleanVar(value=False)
        filter_checkbox = ttk.Checkbutton(filter_enable_frame, text="Enable Filtering",
                                         variable=self.filter_var, command=self.toggle_filter_inputs)
        filter_checkbox.pack(side=tk.LEFT)

        # Filter parameters (initially disabled)
        filter_params_frame = ttk.Frame(settings_tab)
        filter_params_frame.grid(row=10, column=0, sticky=tk.W, padx=(20, 0), pady=5)

        # Max step distance
        step_frame = ttk.Frame(filter_params_frame)
        step_frame.pack(anchor=tk.W, pady=2)
        ttk.Label(step_frame, text="Max step distance (m):").pack(side=tk.LEFT, padx=(0, 5))
        self.max_step_var = tk.StringVar(value="0.03")
        self.max_step_entry = ttk.Entry(step_frame, textvariable=self.max_step_var, width=10, state='disabled')
        self.max_step_entry.pack(side=tk.LEFT)
        ttk.Label(step_frame, text="(outlier threshold. Use about 1.5x pathing speed)", font=("Arial", 9)).pack(side=tk.LEFT, padx=(5, 0))

        # Smooth window
        smooth_frame = ttk.Frame(filter_params_frame)
        smooth_frame.pack(anchor=tk.W, pady=2)
        ttk.Label(smooth_frame, text="Smooth window:").pack(side=tk.LEFT, padx=(0, 5))
        self.smooth_window_var = tk.StringVar(value="71")
        self.smooth_window_entry = ttk.Entry(smooth_frame, textvariable=self.smooth_window_var, width=10, state='disabled')
        self.smooth_window_entry.pack(side=tk.LEFT)
        ttk.Label(smooth_frame, text="(must be odd)", font=("Arial", 9)).pack(side=tk.LEFT, padx=(5, 0))

        # Polynomial order
        poly_frame = ttk.Frame(filter_params_frame)
        poly_frame.pack(anchor=tk.W, pady=2)
        ttk.Label(poly_frame, text="Polynomial order:").pack(side=tk.LEFT, padx=(0, 5))
        self.poly_order_var = tk.StringVar(value="2")
        self.poly_order_entry = ttk.Entry(poly_frame, textvariable=self.poly_order_var, width=10, state='disabled')
        self.poly_order_entry.pack(side=tk.LEFT)
        ttk.Label(poly_frame, text="(Savitzky-Golay filter)", font=("Arial", 9)).pack(side=tk.LEFT, padx=(5, 0))

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0))

        visualize_btn = ttk.Button(button_frame, text="Visualize", command=self.visualize, width=15)
        visualize_btn.pack(side=tk.LEFT, padx=5)

        refresh_btn = ttk.Button(button_frame, text="Refresh List", command=self.populate_csv_files, width=15)
        refresh_btn.pack(side=tk.LEFT, padx=5)

        quit_btn = ttk.Button(button_frame, text="Quit", command=root.quit, width=15)
        quit_btn.pack(side=tk.LEFT, padx=5)

        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Populate initial CSV file list
        self.populate_csv_files()

    def show_help(self):
        """Display help information dialog."""
        help_text = """TRAJECTORY VISUALIZER - QUICK GUIDE

COLOR CODING:
  Green = Simulation data (sim)
  Red = Real hardware data (irl)

MULTI-SELECT:
  Ctrl+Click to select multiple files for comparison.
  Each trajectory pair gets unique colors.

VISUALIZATION:
  • Planned trajectory: First color (blue, green, purple, etc.)
  • Executed trajectory: Second color (red, orange, yellow, etc.)
  • Gray boxes: Physical obstacles
  • Red boxes: Collision zones (obstacle + safety margin)

SETTINGS:
  • Base Offset: Vertical display offset (default: 0.0m)
  • Show Ground Plane: Toggle green reference plane at z=0 (default: off)
  • Show Planned/Executed: Toggle trajectory types for comparison
  • Filtering: Remove outliers and smooth noisy data
        """

        messagebox.showinfo("Help", help_text)

    def toggle_filter_inputs(self):
        """Enable/disable filter parameter inputs based on checkbox."""
        state = 'normal' if self.filter_var.get() else 'disabled'
        self.max_step_entry.config(state=state)
        self.smooth_window_entry.config(state=state)
        self.poly_order_entry.config(state=state)

    def browse_directory(self):
        """Open directory browser dialog."""
        directory = filedialog.askdirectory(initialdir=self.current_dir, title="Select Directory")
        if directory:
            self.current_dir = Path(directory)
            self.dir_var.set(str(self.current_dir))
            self.populate_csv_files()

    def populate_csv_files(self):
        """Find and populate CSV files in the current directory and subdirectories with color coding."""
        # Clear existing items
        for item in self.file_treeview.get_children():
            self.file_treeview.delete(item)

        try:
            # Search for CSV files recursively
            csv_files = sorted(self.current_dir.rglob("*.csv"))

            # Filter out files with "metrics" in the name
            trajectory_files = [f for f in csv_files if "metrics" not in f.stem.lower()]

            if not trajectory_files:
                self.file_treeview.insert('', 'end', text="No trajectory CSV files found in directory", tags=('unknown',))
                return

            # Add files to treeview with color coding based on data source
            for csv_file in trajectory_files:
                relative_path = csv_file.relative_to(self.current_dir)
                display_name = f"{relative_path}"

                # Determine data source from metrics.csv
                data_source = self._get_data_source(csv_file)

                # Add item with appropriate tag for coloring
                self.file_treeview.insert('', 'end', text=display_name, tags=(data_source,))

        except Exception as e:
            messagebox.showerror("Error", f"Failed to scan directory: {e}")

    def _get_data_source(self, trajectory_file):
        """Determine if trajectory is from simulation or real hardware by checking metrics.csv."""
        try:
            # Find metrics.csv in the same directory
            metrics_file = trajectory_file.parent / "metrics.csv"

            if not metrics_file.exists():
                return 'unknown'

            # Extract run number from trajectory filename
            import re
            trajectory_stem = trajectory_file.stem
            match = re.search(r'_(\d+)(?:_sim)?$', trajectory_stem)  # Handle both with and without _sim suffix

            if not match:
                return 'unknown'

            run_number = int(match.group(1))

            # Load metrics and check data_source field
            df = pd.read_csv(metrics_file)

            if run_number <= len(df):
                row_index = run_number - 1
                row = df.iloc[row_index]

                # Check if data_source field exists
                if 'data_source' in row:
                    data_source = str(row['data_source']).lower()
                    if data_source == 'simulation':
                        return 'sim'
                    elif data_source == 'real':
                        return 'real'

            return 'unknown'

        except Exception:
            return 'unknown'

    def on_trajectory_select(self, event):
        """Handle trajectory file selection and load corresponding metrics."""
        selection = self.file_treeview.selection()
        if not selection:
            return

        # If multiple files selected, show multi-select message
        if len(selection) > 1:
            self.metrics_dropdown['values'] = []
            self.metrics_var.set(f"{len(selection)} trajectories selected - ready to visualize")
            self.obstacle_data = None
            return

        selected_item = selection[0]
        selected_file = self.file_treeview.item(selected_item)['text']
        if selected_file in ["No trajectory CSV files found in directory", "No CSV files found in directory"]:
            return

        trajectory_file = self.current_dir / selected_file

        # Extract run number from filename (e.g., "trajectory_1_sim.csv" -> run 1, "trajectory_1.csv" -> run 1)
        trajectory_stem = trajectory_file.stem
        run_number = None

        # Try to extract run number from filename (look for _N or _N_sim pattern)
        import re
        match = re.search(r'_(\d+)(?:_sim)?$', trajectory_stem)
        if match:
            run_number = int(match.group(1))

        # Find metrics.csv in the same directory as the trajectory file
        trajectory_parent = trajectory_file.parent
        metrics_file = trajectory_parent / "metrics.csv"

        if metrics_file.exists():
            try:
                # Load metrics CSV
                df = pd.read_csv(metrics_file)

                # If we found a run number, use that row; otherwise use first row
                if run_number is not None and run_number <= len(df):
                    row_index = run_number - 1  # 0-indexed
                else:
                    row_index = 0

                if row_index < len(df):
                    row = df.iloc[row_index]

                    # Format each metric as a separate dropdown item
                    metrics_list = []
                    for col in df.columns:
                        if isinstance(row[col], (int, float)):
                            metrics_list.append(f"{col}: {row[col]:.3f}")
                        else:
                            metrics_list.append(f"{col}: {row[col]}")

                    self.metrics_dropdown['values'] = metrics_list
                    if metrics_list:
                        self.metrics_var.set(metrics_list[0])
                    else:
                        self.metrics_var.set("No metrics found")

                    # Try to load obstacle data from metrics
                    try:
                        obstacle_cols = ['wall_size_x', 'wall_size_y', 'wall_size_z',
                                       'wall_pos_x', 'wall_pos_y', 'wall_pos_z',
                                       'wall_rot_w', 'wall_rot_x', 'wall_rot_y', 'wall_rot_z']

                        if all(col in row for col in obstacle_cols):
                            safety_margin = float(row["safety_margin"]) if "safety_margin" in row else 0.0
                            size = np.array([row['wall_size_x'], row['wall_size_y'], row['wall_size_z']], dtype=float)
                            padded_size = size + 2.0 * safety_margin  # Inflate to match collision box
                            padded_size = np.maximum(padded_size, 1e-4)  # avoid negative/zero dims
                            self.obstacle_data = [{
                                "size": size,
                                "padded_size": padded_size,
                                "safety_margin": safety_margin,
                                "pos": np.array([row['wall_pos_x'], row['wall_pos_y'], row['wall_pos_z']], dtype=float),
                                "rot": np.array([row['wall_rot_w'], row['wall_rot_x'], row['wall_rot_y'], row['wall_rot_z']], dtype=float)
                            }]
                            print(f"Loaded obstacle data from metrics (with padding): size={size}, padded={padded_size}, pos={self.obstacle_data[0]['pos']}, safety_margin={safety_margin}")
                        else:
                            print("Obstacle data not found in metrics (using defaults)")
                            self.obstacle_data = None
                    except Exception as e:
                        print(f"Warning: Could not parse obstacle data from metrics: {e}")
                        self.obstacle_data = None
                else:
                    self.metrics_var.set("No metrics found for this run")
                    self.metrics_dropdown['values'] = []
                    self.obstacle_data = None

            except Exception as e:
                self.metrics_dropdown['values'] = []
                self.metrics_var.set(f"Error loading metrics: {e}")
                self.obstacle_data = None
        else:
            self.metrics_dropdown['values'] = []
            self.metrics_var.set("No metrics.csv file found in folder")
            self.obstacle_data = None

    def visualize(self):
        """Visualize the selected trajectory file(s) - supports multi-select."""
        selection = self.file_treeview.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select one or more CSV files to visualize")
            return

        # Get base offset
        try:
            base_offset = float(self.offset_var.get())
        except ValueError:
            messagebox.showerror("Error", "Base offset must be a valid number")
            return

        # Get filtering settings
        apply_filtering = self.filter_var.get()

        # Get filter parameters if filtering is enabled
        if apply_filtering:
            try:
                max_step_distance = float(self.max_step_var.get())
                smooth_window = int(self.smooth_window_var.get())
                polyorder = int(self.poly_order_var.get())

                # Validate smooth window is odd
                if smooth_window % 2 == 0:
                    messagebox.showerror("Error", "Smooth window must be an odd number")
                    return
            except ValueError:
                messagebox.showerror("Error", "Filter parameters must be valid numbers")
                return
        else:
            # Use defaults if filtering is disabled (won't be used anyway)
            max_step_distance = 0.03
            smooth_window = 71
            polyorder = 2

        # Initialize visualizer
        visualizer = TrajectoryVisualizer(excavator_base_offset=base_offset)

        # Collect all trajectory data
        trajectory_data_list = []
        obstacle_data = None
        all_obstacle_configs = []  # Track obstacle configs for consistency check

        try:
            for selected_item in selection:
                selected_file = self.file_treeview.item(selected_item)['text']
                if selected_file in ["No CSV files found in directory", "No trajectory CSV files found in directory"]:
                    continue

                trajectory_file = self.current_dir / selected_file

                # Validate file exists
                if not trajectory_file.exists():
                    messagebox.showerror("Error", f"File not found: {trajectory_file}")
                    continue

                # Load trajectory data with custom filter parameters
                planned_traj, executed_traj = visualizer.load_trajectory_data(
                    str(trajectory_file),
                    apply_filtering=apply_filtering,
                    max_step_distance=max_step_distance,
                    smooth_window=smooth_window,
                    polyorder=polyorder
                )

                # Get data source from metrics
                data_source = self._get_data_source(trajectory_file)

                # Create short name from filename (remove path and extension)
                short_name = trajectory_file.stem

                # Add to trajectory list
                trajectory_data_list.append({
                    'planned': planned_traj,
                    'executed': executed_traj,
                    'name': short_name,
                    'source': data_source
                })

                # Load obstacle configuration from each trajectory for consistency checking
                trajectory_parent = trajectory_file.parent
                metrics_file = trajectory_parent / "metrics.csv"

                if metrics_file.exists():
                    try:
                        # Extract run number from trajectory filename
                        import re
                        trajectory_stem = trajectory_file.stem
                        match = re.search(r'_(\d+)(?:_sim)?$', trajectory_stem)
                        row_index = int(match.group(1)) - 1 if match else 0

                        # Load metrics
                        df = pd.read_csv(metrics_file)
                        if row_index < len(df):
                            row = df.iloc[row_index]

                            # Check for obstacle columns
                            obstacle_cols = ['wall_size_x', 'wall_size_y', 'wall_size_z',
                                           'wall_pos_x', 'wall_pos_y', 'wall_pos_z',
                                           'wall_rot_w', 'wall_rot_x', 'wall_rot_y', 'wall_rot_z']

                            if all(col in row for col in obstacle_cols):
                                config = {
                                    'file': short_name,
                                    'size': (row['wall_size_x'], row['wall_size_y'], row['wall_size_z']),
                                    'pos': (row['wall_pos_x'], row['wall_pos_y'], row['wall_pos_z']),
                                    'rot': (row['wall_rot_w'], row['wall_rot_x'], row['wall_rot_y'], row['wall_rot_z']),
                                    'safety_margin': float(row["safety_margin"]) if "safety_margin" in row else 0.0
                                }
                                all_obstacle_configs.append(config)

                                # Store first trajectory's obstacle data for visualization
                                if obstacle_data is None:
                                    safety_margin = config['safety_margin']
                                    size = np.array(config['size'], dtype=float)
                                    padded_size = size + 2.0 * safety_margin
                                    padded_size = np.maximum(padded_size, 1e-4)
                                    obstacle_data = [{
                                        "size": size,
                                        "padded_size": padded_size,
                                        "safety_margin": safety_margin,
                                        "pos": np.array(config['pos'], dtype=float),
                                        "rot": np.array(config['rot'], dtype=float)
                                    }]
                                    print(f"Using obstacle data from first trajectory: {short_name}")
                    except Exception as e:
                        print(f"Warning: Could not load obstacle data from {short_name}: {e}")

            if not trajectory_data_list:
                messagebox.showwarning("No Data", "No valid trajectories loaded")
                return

            # Check for obstacle/goal pose consistency across trajectories
            if len(all_obstacle_configs) > 1:
                first_config = all_obstacle_configs[0]
                differences_found = []

                for config in all_obstacle_configs[1:]:
                    # Check size
                    if not np.allclose(first_config['size'], config['size'], atol=1e-6):
                        differences_found.append(f"  • Wall size differs: {first_config['file']} vs {config['file']}")
                    # Check position
                    if not np.allclose(first_config['pos'], config['pos'], atol=1e-6):
                        differences_found.append(f"  • Wall position differs: {first_config['file']} vs {config['file']}")
                    # Check rotation
                    if not np.allclose(first_config['rot'], config['rot'], atol=1e-6):
                        differences_found.append(f"  • Wall rotation differs: {first_config['file']} vs {config['file']}")
                    # Check safety margin
                    if not np.isclose(first_config['safety_margin'], config['safety_margin'], atol=1e-6):
                        differences_found.append(f"  • Safety margin differs: {first_config['file']} vs {config['file']}")

                if differences_found:
                    warning_msg = (
                        "⚠️ Obstacle configurations differ between selected trajectories:\n\n" +
                        "\n".join(differences_found[:5]) +  # Show max 5 differences
                        (f"\n  ... and {len(differences_found)-5} more" if len(differences_found) > 5 else "") +
                        f"\n\nUsing obstacle from: {first_config['file']}\n\nContinue with visualization?"
                    )

                    if not messagebox.askyesno("Obstacle Configuration Mismatch", warning_msg):
                        return  # User chose not to continue
                    print(f"User acknowledged obstacle differences, continuing with {first_config['file']} configuration")

            if not trajectory_data_list:
                messagebox.showwarning("No Data", "No valid trajectories loaded")
                return

            # Use empty obstacle data if none loaded
            if obstacle_data is None:
                obstacle_data = []
                print("No obstacle data available; skipping obstacle render")

            # Generate title based on number of trajectories
            if len(trajectory_data_list) == 1:
                algorithm = detect_algorithm_from_filename(trajectory_data_list[0]['name'])
                title = f"{algorithm} Trajectory Visualization"
            else:
                title = f"Multi-Trajectory Comparison ({len(trajectory_data_list)} trajectories)"

            # Get visualization settings
            show_ground_plane = self.show_ground_plane_var.get()
            show_planned = self.show_planned_var.get()
            show_executed = self.show_executed_var.get()
            show_obstacles = self.show_obstacles_var.get()
            show_collision = self.show_collision_var.get()

            # Validate that at least one trajectory type is selected
            if not show_planned and not show_executed:
                messagebox.showwarning("No Trajectories Selected",
                                      "Please enable at least one trajectory type (Planned or Executed)")
                return

            # Create visualization with all trajectories
            visualizer.plot_trajectories(trajectory_data_list, obstacle_data, title,
                                        show_ground_plane, show_planned, show_executed,
                                        show_obstacles, show_collision)

        except Exception as e:
            messagebox.showerror("Visualization Error", f"Failed to visualize trajectory:\n{e}")


def main():
    """Main function - launches GUI mode by default."""
    # Always launch GUI (great UI!)
    root = tk.Tk()
    app = TrajectoryVisualizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    """
    Trajectory Visualizer - GUI Mode

    Usage:
        python trajectory_visualization.py

    Features:
        • Browse and select trajectory CSV files
        • Multi-select support (Ctrl+Click) for trajectory comparison
        • Automatic obstacle detection from metrics
        • Consistency checking when comparing multiple trajectories
        • Color-coded (sim) vs (irl) data sources
    """
    main()
