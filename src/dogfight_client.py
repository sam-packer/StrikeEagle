import json
from src import socket_lib


def _encode(command, args=None):
    if args is None:
        args = {}
    return str.encode(json.dumps({"command": command, "args": args}))


def _send(command, args=None):
    socket_lib.send_message(_encode(command, args))


def send_batch(commands):
    """Send multiple fire-and-forget commands in one TCP write.

    commands: list of (command_name, args_dict) tuples
    """
    socket_lib.send_messages_batch([_encode(cmd, args) for cmd, args in commands])


def _send_recv(command, args=None):
    _send(command, args)
    answer = socket_lib.get_answer()
    if answer is None:
        raise ConnectionError(f"No response from sandbox for command: {command}")
    return json.loads(answer.decode())


# Connection
def connect(host, port):
    socket_lib.connect_socket(host, port)


def disconnect():
    socket_lib.close_socket()


# Global / scene control
def disable_log():
    _send("DISABLE_LOG")


def enable_log():
    _send("ENABLE_LOG")


def get_running():
    return _send_recv("GET_RUNNING")


def set_timestep(t):
    _send("SET_TIMESTEP", {"timestep": t})


def get_timestep():
    return _send_recv("GET_TIMESTEP")


def step(plane_id, pitch, roll, yaw, thrust):
    """Apply controls + advance sim + return plane state in one round-trip."""
    return _send_recv("STEP", {
        "plane_id": plane_id,
        "pitch": pitch,
        "roll": roll,
        "yaw": yaw,
        "thrust": thrust,
    })


def step_n(plane_id, pitch, roll, yaw, thrust, n):
    """Apply controls + advance sim *n* frames + return plane state in one round-trip."""
    return _send_recv("STEP_N", {
        "plane_id": plane_id,
        "pitch": pitch,
        "roll": roll,
        "yaw": yaw,
        "thrust": thrust,
        "n": n,
    })


def set_renderless_mode(flag: bool):
    _send("SET_RENDERLESS_MODE", {"flag": flag})


def set_client_update_mode(flag: bool):
    _send("SET_CLIENT_UPDATE_MODE", {"flag": flag})


def set_display_radar_in_renderless_mode(flag: bool):
    _send("SET_DISPLAY_RADAR_IN_RENDERLESS_MODE", {"flag": flag})


def update_scene():
    _send("UPDATE_SCENE")


# Machine (generic) commands
def get_mobile_parts_list(machine_id):
    return _send_recv("GET_MOBILE_PARTS_LIST", {"machine_id": machine_id})


def set_machine_custom_physics_mode(machine_id, flag: bool):
    _send("SET_MACHINE_CUSTOM_PHYSICS_MODE", {"machine_id": machine_id, "flag": flag})


def get_targets_list(machine_id):
    return _send_recv("GET_TARGETS_LIST", {"machine_id": machine_id})


def get_target_idx(machine_id):
    return _send_recv("GET_TARGET_IDX", {"machine_id": machine_id})


def set_target_id(machine_id, target_id):
    _send("SET_TARGET_ID", {"machine_id": machine_id, "target_id": target_id})


def get_health(machine_id):
    return _send_recv("GET_HEALTH", {"machine_id": machine_id})


def set_health(machine_id, level):
    _send("SET_HEALTH", {"machine_id": machine_id, "health_level": level})


def get_machine_missiles_list(machine_id):
    return _send_recv("GET_MACHINE_MISSILES_LIST", {"machine_id": machine_id})


def get_machine_gun_state(machine_id):
    return _send_recv("GET_MACHINE_GUN_STATE", {"machine_id": machine_id})


def activate_machine_gun(machine_id):
    _send("ACTIVATE_MACHINE_GUN", {"machine_id": machine_id})


def deactivate_machine_gun(machine_id):
    _send("DEACTIVATE_MACHINE_GUN", {"machine_id": machine_id})


def get_missiles_device_slots_state(machine_id):
    return _send_recv("GET_MISSILESDEVICE_SLOTS_STATE", {"machine_id": machine_id})


def fire_missile(machine_id, slot_id, target_id=None):
    """Fire missile from slot. If target_id is provided, forces target lock before firing."""
    args = {"machine_id": machine_id, "slot_id": slot_id}
    if target_id is not None:
        args["target_id"] = target_id
    _send("FIRE_MISSILE", args)


def rearm_machine(machine_id):
    _send("REARM_MACHINE", {"machine_id": machine_id})


def reset_machine(machine_id):
    _send("RESET_MACHINE", {"machine_id": machine_id})


def reset_machine_matrix(machine_id, x, y, z, rx, ry, rz):
    _send(
        "RESET_MACHINE_MATRIX",
        {"machine_id": machine_id, "position": [x, y, z], "rotation": [rx, ry, rz]},
    )


def activate_autopilot(machine_id):
    _send("ACTIVATE_AUTOPILOT", {"machine_id": machine_id})


def deactivate_autopilot(machine_id):
    _send("DEACTIVATE_AUTOPILOT", {"machine_id": machine_id})


def activate_IA(machine_id):
    _send("ACTIVATE_IA", {"machine_id": machine_id})


def deactivate_IA(machine_id):
    _send("DEACTIVATE_IA", {"machine_id": machine_id})


def is_user_control_activated(machine_id):
    return _send_recv("IS_USER_CONTROL_ACTIVATED", {"machine_id": machine_id})


def activate_user_control(machine_id):
    _send("ACTIVATE_USER_CONTROL", {"machine_id": machine_id})


def deactivate_user_control(machine_id):
    _send("DEACTIVATE_USER_CONTROL", {"machine_id": machine_id})


# Aircraft-specific commands
def get_planes_list():
    return _send_recv("GET_PLANESLIST")


def get_plane_state(plane_id):
    return _send_recv("GET_PLANE_STATE", {"plane_id": plane_id})


def get_plane_thrust(plane_id):
    return _send_recv("GET_PLANE_THRUST", {"plane_id": plane_id})


def set_plane_thrust(plane_id, level):
    _send("SET_PLANE_THRUST", {"plane_id": plane_id, "thrust_level": level})


def set_plane_linear_speed(plane_id, speed):
    _send("SET_PLANE_LINEAR_SPEED", {"plane_id": plane_id, "linear_speed": speed})


def set_plane_brake(plane_id, level):
    _send("SET_PLANE_BRAKE", {"plane_id": plane_id, "brake_level": level})


def set_plane_flaps(plane_id, level):
    _send("SET_PLANE_FLAPS", {"plane_id": plane_id, "flaps_level": level})


def activate_post_combustion(plane_id):
    _send("ACTIVATE_PC", {"plane_id": plane_id})


def deactivate_post_combustion(plane_id):
    _send("DEACTIVATE_PC", {"plane_id": plane_id})


def set_plane_pitch(plane_id, level):
    _send("SET_PLANE_PITCH", {"plane_id": plane_id, "pitch_level": level})


def set_plane_roll(plane_id, level):
    _send("SET_PLANE_ROLL", {"plane_id": plane_id, "roll_level": level})


def set_plane_yaw(plane_id, level):
    _send("SET_PLANE_YAW", {"plane_id": plane_id, "yaw_level": level})


def stabilize_plane(plane_id):
    _send("STABILIZE_PLANE", {"plane_id": plane_id})


def deploy_gear(plane_id):
    _send("DEPLOY_GEAR", {"plane_id": plane_id})


def retract_gear(plane_id):
    _send("RETRACT_GEAR", {"plane_id": plane_id})


def record_plane_start_state(plane_id):
    _send("RECORD_PLANE_START_STATE", {"plane_id": plane_id})


def set_plane_autopilot_heading(plane_id, heading):
    _send("SET_PLANE_AUTOPILOT_HEADING", {"plane_id": plane_id, "ap_heading": heading})


def set_plane_autopilot_speed(plane_id, speed):
    _send("SET_PLANE_AUTOPILOT_SPEED", {"plane_id": plane_id, "ap_speed": speed})


def set_plane_autopilot_altitude(plane_id, altitude):
    _send("SET_PLANE_AUTOPILOT_ALTITUDE", {"plane_id": plane_id, "ap_altitude": altitude})


# Missile launchers
def get_missile_launchers_list():
    return _send_recv("GET_MISSILE_LAUNCHERS_LIST")


def get_missile_launcher_state(machine_id):
    return _send_recv("GET_MISSILE_LAUNCHER_STATE", {"machine_id": machine_id})


# Missiles
def get_missiles_list():
    return _send_recv("GET_MISSILESLIST")


def get_missile_state(missile_id):
    return _send_recv("GET_MISSILE_STATE", {"missile_id": missile_id})


def set_missile_target(missile_id, target_id):
    _send("SET_MISSILE_TARGET", {"missile_id": missile_id, "target_id": target_id})


def set_missile_life_delay(missile_id, life_delay):
    _send("SET_MISSILE_LIFE_DELAY", {"missile_id": missile_id, "life_delay": life_delay})


def get_missile_targets_list(missile_id):
    return _send_recv("GET_MISSILE_TARGETS_LIST", {"missile_id": missile_id})


def set_missile_thrust_force(missile_id, thrust_force):
    _send("SET_MISSILE_THRUST_FORCE", {"missile_id": missile_id, "thrust_force": thrust_force})


def set_missile_angular_frictions(missile_id, x, y, z):
    _send(
        "SET_MISSILE_ANGULAR_FRICTIONS",
        {"missile_id": missile_id, "angular_frictions": [x, y, z]},
    )


def set_missile_drag_coefficients(missile_id, x, y, z):
    _send(
        "SET_MISSILE_DRAG_COEFFICIENTS",
        {"missile_id": missile_id, "drag_coeff": [x, y, z]},
    )


# Carriers
def get_carriers_list():
    return _send_recv("GET_CARRIERS_LIST")


def get_carrier_deck_parameters(carrier_id):
    return _send_recv("GET_CARRIER_DECK_PARAMETERS", {"carrier_id": carrier_id})


# Camera / view
def set_camera_track(machine_id):
    """Set camera to follow a specific aircraft (also sets audio listener)."""
    _send("SET_CAMERA_TRACK", {"machine_id": machine_id})


def set_track_view(view):
    """Set camera angle: 'back', 'front', 'left', 'right', or 'top'."""
    _send("SET_TRACK_VIEW", {"view": view})
