from omni.isaac.core import SimulationContext
from omni.isaac.core.objects import DynamicCuboid, DynamicCylinder
from omni.isaac.core.prims import XFormPrim, GeometryPrim
from pxr import Gf, UsdGeom, Sdf
import numpy as np

class HydraulicCylinderHelper:
    def __init__(self, stage):
        """
        Initialize the hydraulic cylinder helper
        
        Args:
            stage: Current USD stage
        """
        self.stage = stage
        
    def _get_xform_world_position(self, xform_path):
        """Get world position of an xform"""
        xform = UsdGeom.Xform(self.stage.GetPrimAtPath(xform_path))
        if not xform:
            raise ValueError(f"XForm not found at path: {xform_path}")
        
        # Get world transform matrix
        world_transform = xform.ComputeLocalToWorldTransform(0.0)
        # Extract position
        return (world_transform.GetTranslation().GetArray())
    
    def _calculate_cylinder_transform(self, start_pos, end_pos):
        """Calculate position, rotation and length for cylinder"""
        # Calculate direction vector
        direction = np.array(end_pos) - np.array(start_pos)
        length = np.linalg.norm(direction)
        
        # Calculate center position
        center = (np.array(start_pos) + np.array(end_pos)) / 2
        
        # Calculate rotation to align with direction
        # Default cylinder axis is along Z
        up_vector = np.array([0, 0, 1])
        direction_normalized = direction / length
        
        # Calculate rotation axis and angle
        rotation_axis = np.cross(up_vector, direction_normalized)
        rotation_axis_length = np.linalg.norm(rotation_axis)
        
        if rotation_axis_length > 1e-6:  # If not parallel to up_vector
            rotation_axis = rotation_axis / rotation_axis_length
            angle = np.arccos(np.dot(up_vector, direction_normalized))
            
            # Convert to quaternion
            quat = Gf.Rotation(rotation_axis, np.rad2deg(angle)).GetQuat()
        else:
            # Handle case when vectors are parallel or anti-parallel
            quat = Gf.Quaternion(1, 0, 0, 0)
            if np.dot(up_vector, direction_normalized) < 0:
                quat = Gf.Quaternion(0, 1, 0, 0)  # 180-degree rotation around X
                
        return center, quat, length
    
    def create_cylinder_between_points(
        self,
        prim_path: str,
        start_point_path: str,
        end_point_path: str,
        cylinder_radius=0.05,
        piston_radius=0.04,
        damping=100,
        stiffness=1000,
        max_force=1000,
        max_velocity=1.0,
        color=(0.7, 0.7, 0.7)
    ):
        """
        Creates a hydraulic cylinder between two XForm points in the scene
        
        Args:
            prim_path: Base path for the cylinder assembly
            start_point_path: Path to the XForm marking cylinder base point
            end_point_path: Path to the XForm marking cylinder end point
            cylinder_radius: Radius of outer cylinder
            piston_radius: Radius of inner piston
            damping: Joint damping coefficient
            stiffness: Joint stiffness
            max_force: Maximum force the cylinder can apply
            max_velocity: Maximum extension/retraction velocity
            color: RGB color tuple for the cylinder
        
        Returns:
            Dictionary containing references to created components
        """
        # Get positions from XForms
        start_pos = self._get_xform_world_position(start_point_path)
        end_pos = self._get_xform_world_position(end_point_path)
        
        # Calculate transform
        center, rotation, total_length = self._calculate_cylinder_transform(start_pos, end_pos)
        
        # Create assembly group
        assembly_prim = UsdGeom.Xform.Define(self.stage, prim_path)
        
        # Create the cylinder housing (shorter than total length to allow for movement)
        cylinder_length = total_length * 0.6  # 60% of total length
        housing = DynamicCylinder(
            prim_path=f"{prim_path}/housing",
            name="cylinder_housing",
            position=center,
            orientation=rotation,
            radius=cylinder_radius,
            height=cylinder_length,
            color=color,
        )
        
        # Create the piston rod
        piston_length = total_length * 0.8  # 80% of total length
        piston = DynamicCylinder(
            prim_path=f"{prim_path}/piston",
            name="piston_rod",
            position=center,
            orientation=rotation,
            radius=piston_radius,
            height=piston_length,
            color=color,
        )
        
        # Create prismatic joint
        joint = XFormPrim(
            prim_path=f"{prim_path}/joint",
            name="hydraulic_joint"
        )
        
        # Configure joint properties
        joint.CreateAttribute("joint:type", "string", "prismatic")
        joint.CreateAttribute("joint:lowerLimit", "float", -cylinder_length/3)
        joint.CreateAttribute("joint:upperLimit", "float", cylinder_length/3)
        joint.CreateAttribute("joint:maxForce", "float", max_force)
        joint.CreateAttribute("joint:maxVelocity", "float", max_velocity)
        joint.CreateAttribute("joint:damping", "float", damping)
        joint.CreateAttribute("joint:stiffness", "float", stiffness)
        
        # Set joint axis aligned with cylinder direction
        joint.CreateAttribute("joint:axis", "float3", (0, 0, 1))
        
        # Store references to start and end points
        self._store_endpoint_references(prim_path, start_point_path, end_point_path)
        
        return {
            "housing": housing,
            "piston": piston,
            "joint": joint,
            "start_point": start_point_path,
            "end_point": end_point_path,
            "length": total_length
        }
    
    def _store_endpoint_references(self, prim_path, start_point_path, end_point_path):
        """Store references to endpoint XForms for later use"""
        assembly_prim = self.stage.GetPrimAtPath(prim_path)
        assembly_prim.CreateAttribute("customData:startPoint", Sdf.ValueTypeNames.String).Set(start_point_path)
        assembly_prim.CreateAttribute("customData:endPoint", Sdf.ValueTypeNames.String).Set(end_point_path)

# Example usage:
"""
# Initialize simulation
sim = SimulationContext()
stage = sim.get_stage()

# Create helper
cylinder_helper = HydraulicCylinderHelper(stage)

# Create cylinder between two existing XForms in your scene
cylinder = cylinder_helper.create_cylinder_between_points(
    prim_path="/World/hydraulic_cylinder_1",
    start_point_path="/World/attachment_point_1",
    end_point_path="/World/attachment_point_2",
    cylinder_radius=0.03,
    color=(0.7, 0.7, 0.8)
)

# Control the cylinder
cylinder["joint"].apply_force(500)  # Extend with 500N force
# or
cylinder["joint"].set_target_position(0.2)  # Extend by 0.2m

# Access cylinder properties
print(f"Cylinder length: {cylinder['length']}")
print(f"Start point: {cylinder['start_point']}")
print(f"End point: {cylinder['end_point']}")
"""