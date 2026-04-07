"""Minimal reproduction: find exactly which command kills the connection."""

import time
import sys
sys.path.insert(0, ".")
from src import dogfight_client as df
from src.net_utils import get_lan_ip

host = get_lan_ip()
port = 50888

def try_cmd(label, fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        print(f"  OK: {label}")
        return result
    except Exception as e:
        print(f"  FAIL: {label} -> {e}")
        sys.exit(1)

print(f"Connecting to {host}:{port}...")
df.connect(host, port)
time.sleep(2)
print("Connected.\n")

planes = try_cmd("get_planes_list", df.get_planes_list)
print(f"  Planes: {planes}\n")

ally = "ally_1"
enemy = "ennemy_1"

print("=== Setting modes ===")
try_cmd("disable_log", df.disable_log)
try_cmd("set_client_update_mode(True)", df.set_client_update_mode, True)
try_cmd("set_renderless_mode(True)", df.set_renderless_mode, True)
time.sleep(1)
r = try_cmd("get_running", df.get_running)
print(f"  running={r}\n")

print("=== Phase A: Reset ally only (like Phase 5) ===")
try_cmd("reset_machine(ally)", df.reset_machine, ally)
try_cmd("reset_machine_matrix(ally)", df.reset_machine_matrix, ally, 0, 4000, 0, 0, 0, 0)
try_cmd("set_plane_thrust(ally, 1)", df.set_plane_thrust, ally, 1.0)
try_cmd("set_plane_linear_speed(ally, 300)", df.set_plane_linear_speed, ally, 300)
try_cmd("retract_gear(ally)", df.retract_gear, ally)
try_cmd("get_plane_state(ally) [sync]", df.get_plane_state, ally)
try_cmd("update_scene [1]", df.update_scene)
try_cmd("get_plane_state(ally) [after update]", df.get_plane_state, ally)
print("  Phase A passed!\n")

print("=== Phase B: Add enemy reset ===")
try_cmd("reset_machine(enemy)", df.reset_machine, enemy)
try_cmd("get_plane_state(ally) [sync]", df.get_plane_state, ally)
try_cmd("update_scene [2]", df.update_scene)
try_cmd("get_plane_state(ally) [after update]", df.get_plane_state, ally)
print("  Phase B passed!\n")

print("=== Phase C: Position enemy ===")
try_cmd("reset_machine_matrix(enemy)", df.reset_machine_matrix, enemy, 0, 4200, -5000, 0, 0, 0)
try_cmd("set_plane_thrust(enemy, 0.8)", df.set_plane_thrust, enemy, 0.8)
try_cmd("set_plane_linear_speed(enemy, 250)", df.set_plane_linear_speed, enemy, 250)
try_cmd("retract_gear(enemy)", df.retract_gear, enemy)
try_cmd("get_plane_state(ally) [sync]", df.get_plane_state, ally)
try_cmd("update_scene [3]", df.update_scene)
try_cmd("get_plane_state(ally) [after update]", df.get_plane_state, ally)
print("  Phase C passed!\n")

print("=== Phase D: Target + health + rearm ===")
try_cmd("set_target_id(enemy, ally)", df.set_target_id, enemy, ally)
try_cmd("get_plane_state(ally) [sync]", df.get_plane_state, ally)
try_cmd("update_scene [4]", df.update_scene)
print("  D1: target OK")

try_cmd("set_health(ally, 1.0)", df.set_health, ally, 1.0)
try_cmd("set_health(enemy, 1.0)", df.set_health, enemy, 1.0)
try_cmd("get_plane_state(ally) [sync]", df.get_plane_state, ally)
try_cmd("update_scene [5]", df.update_scene)
print("  D2: health OK")

try_cmd("rearm_machine(enemy)", df.rearm_machine, enemy)
try_cmd("get_plane_state(ally) [sync]", df.get_plane_state, ally)
try_cmd("update_scene [6]", df.update_scene)
print("  D3: rearm OK")
print("  Phase D passed!\n")

print("=== Phase E: Simulate step() ===")
try_cmd("set_plane_pitch(ally, 0.1)", df.set_plane_pitch, ally, 0.1)
try_cmd("set_plane_roll(ally, -0.2)", df.set_plane_roll, ally, -0.2)
try_cmd("set_plane_yaw(ally, 0.0)", df.set_plane_yaw, ally, 0.0)
try_cmd("set_plane_thrust(ally, 0.8)", df.set_plane_thrust, ally, 0.8)
try_cmd("set_plane_pitch(enemy, 0)", df.set_plane_pitch, enemy, 0)
try_cmd("set_plane_roll(enemy, 0)", df.set_plane_roll, enemy, 0)
try_cmd("set_plane_yaw(enemy, 0)", df.set_plane_yaw, enemy, 0)
try_cmd("get_plane_state(ally) [sync]", df.get_plane_state, ally)
try_cmd("update_scene [step]", df.update_scene)
state = try_cmd("get_plane_state(ally) [obs]", df.get_plane_state, ally)
print(f"  alt={state['altitude']:.1f}m, speed={state['linear_speed']:.1f}m/s")
print("  Phase E passed!\n")

print("=== ALL PHASES PASSED ===")
df.disconnect()
