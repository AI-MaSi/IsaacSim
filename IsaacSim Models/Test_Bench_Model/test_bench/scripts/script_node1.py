# placeholder for mangling the inputs to (kinda) mimic real hydraulic system

# TODO (just an idea):
# create splines from IRL driving and from solidworks simulations
# interpolate and normalize
# use here as a filter

def setup(db: og.Database):
    # dont know if this is necessary
    db.inputs.articulation_feedback = [0.0, 0.0, 0.0]
    db.outputs.control_output = [0.0, 0.0, 0.0]
    pass


def compute(db):


    db.outputs.control_output = db.inputs.control_input # placeholder
    # db.outputs.control_output = db.inputs.control_input * apply_linkage_spline(db.inputs.articulation_feedback) * apply_hydraulic_multipliers(db.inputs.control_input)

    return True


def cleanup(db: og.Database):
    db.inputs.articulation_feedback = [0.0, 0.0, 0.0]
    db.outputs.control_output = [0.0, 0.0, 0.0]
    pass


def rad_to_deg(feedback_array):
    angle_offsets = db.inputs.angle_offsets
    # Convert radians to degrees using formula: degrees = radians * (180/π)
    conversion_factor = 180.0 / 3.14159265359

    # Convert each element of the array
    converted_array = [
        feedback_array[0] * conversion_factor,
        feedback_array[1] * conversion_factor,
        feedback_array[2] * conversion_factor
    ]

    # If you need to offset the values (e.g., if 0 degrees should be at 90 degrees)
    offset_array = [
        converted_array[0] + angle_offset[0],
        converted_array[1] + angle_offset[1],
        converted_array[2] + angle_offset[2]
    ]

    return offset_array


def apply_linkage_spline(feedback_array):
    """
    Adjusts control based on current geometric configuration
    feedback_array: Current joint angles in radians
    Returns: Multiplier for each joint's movement rate
    """
    input_rate_multipliers = [1.0, 1.0, 1.0]  # placeholder

    # For each joint:
    # 1. Convert angle to lookup parameter
    # 2. Find neighboring points in spline data
    # 3. Interpolate multiplier
    # This accounts for mechanical advantage changing with position

    return input_rate_multipliers


def apply_hydraulic_multipliers(input_array):
    """
    Simulates hydraulic system characteristics
    input_array: Requested movement rates
    Returns: Modified rates accounting for flow behavior
    """
    hydraulic_multipliers = [1.0, 1.0, 1.0]  # placeholder

    # Consider:
    # 1. Flow sharing between actuators
    # 2. Valve deadbands
    # 3. Flow vs pressure characteristics
    # 4. System pressure effects

    return hydraulic_multipliers
