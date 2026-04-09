# Copyright (C) 2018-2021 Eric Kernin, NWNC HARFANG.

import harfang as hg
from math import radians, degrees, pi, sqrt, exp, floor, acos, asin, atan2
import MathsSupp as ms
from MathsSupp import *
import tools as tools
from overlays import *

air_density0 = 1.225 #  sea level standard atmospheric pressure, 101325 Pa
F_gravity = hg.Vec3(0, -9.8, 0)

scene = None
scene_physics = None
water_level = 0


terrain_heightmap = None
# the bounding box of the actual terrain island geometry, is 36000 x 708 x 11300 m
# the lower point of the heightmap is below the sea level, at -296.87 m
# the highest point of the mountain chains in the island is ~703 m
terrain_position = hg.Vec3(-24896, -296.87, 9443)
terrain_scale = hg.Vec3(41480, 1000, 19587)
map_bounds = hg.Vec2(0, 255)


def init_physics(scn, scn_physics, terrain_heightmap_file, p_terrain_pos, p_terrain_scale, p_map_bounds):
	global scene, scene_physics, terrain_heightmap, terrain_position, terrain_scale, map_bounds
	scene = scn
	scene_physics = scn_physics
	terrain_heightmap = hg.Picture()
	terrain_position = p_terrain_pos
	terrain_scale = p_terrain_scale
	map_bounds = p_map_bounds
	hg.LoadPicture(terrain_heightmap, terrain_heightmap_file)


def get_terrain_Y(pos: hg.Vec3):
	global terrain_position, terrain_scale, terrain_heightmap, map_bounds
	pos2 = hg.Vec2((pos.x - terrain_position.x) / terrain_scale.x, 1 - (pos.z - terrain_position.z) / terrain_scale.z)
	return get_map_altitude(pos2), get_terrain_XZ(pos2)


def get_map_altitude(pos2d):
	global terrain_position, terrain_scale, terrain_heightmap, map_bounds
	a = (tools.get_pixel_bilinear(terrain_heightmap, pos2d).r * 255 - map_bounds.x) / (map_bounds.y - map_bounds.x)
	a = max(water_level, a * terrain_scale.y + terrain_position.y)
	return a


def get_terrain_XZ(pos2d):
	w = terrain_heightmap.GetWidth()
	h = terrain_heightmap.GetHeight()
	f = 1 / max(w, h)
	xd = hg.Vec2(f, 0)
	zd = hg.Vec2(0, f)
	return hg.Normalize(hg.Vec3(get_map_altitude(pos2d - xd) - get_map_altitude(pos2d + xd), 2 * f, get_map_altitude(pos2d - zd) - get_map_altitude(pos2d + zd)))


def _compute_atmosphere_temp(altitude):
	"""
	Internal function to compute atmospheric temperature according to altitude. Different layers have
	different temperature gradients, therefore the calculation is branched.
	Model is taken from ICAO DOC 7488: Manual of ICAO Standard Atmosphere.

	:param altitude: altitude in meters.
	:return: temperature in Kelvin.
	"""

	#Gradients are Kelvin/km.
	if altitude < 11e3:
		temperature_gradient = -6.5 #Kelvin per km.
		reference_temp = 288.15 #Temperature at sea level.
		altitude_diff = altitude - 0
	else:
		temperature_gradient = 0
		reference_temp = 216.65 #Temperature at 11km altitude.
		altitude_diff = altitude - 11e3

	return reference_temp + temperature_gradient*(altitude_diff / 1000)



def compute_atmosphere_density(altitude):
	# Barometric formula
	# temperature_K : based on ICAO Standard Atmosphere
	temperature_K = _compute_atmosphere_temp(altitude)
	R = 8.3144621  # ideal (universal) gas constant, 8.31446 J/(mol·K)
	M = 0.0289652  # molar mass of dry air, 0.0289652 kg/mol
	g = 9.80665  # earth-surface gravitational acceleration, 9.80665 m/s2
	d = air_density0 * exp(-altitude / (R * temperature_K / (M * g)))
	return d


def _clamp(v, lo, hi):
	return max(lo, min(hi, v))


def _lerp(a, b, t):
	return a + (b - a) * t


def _smoothstep(edge0, edge1, x):
	if edge1 <= edge0:
		return 1.0 if x >= edge1 else 0.0
	return ms.MathsSupp.smoothstep(edge0, edge1, x)


def update_collisions(matrix: hg.Mat4, collisions_object, collisions_raycasts):
	rays_hits = []

	for collision_ray in collisions_raycasts:
		ray_hits = {"name": collision_ray["name"], "hits": []}
		c_pos = matrix * collision_ray["position"]
		c_dir = matrix * (collision_ray["position"] + collision_ray["direction"])
		rc_len = hg.Len(collision_ray["direction"])

		hit = scene_physics.RaycastFirstHit(scene, c_pos, c_dir)

		if 0 < hit.t < rc_len:
			if not collisions_object.test_collision(hit.node):
				ray_hits["hits"].append(hit)
		rays_hits.append(ray_hits)

	terrain_alt, terrain_nrm = get_terrain_Y(hg.GetT(matrix))

	return rays_hits, terrain_alt, terrain_nrm


def update_physics(matrix, collisions_object, physics_parameters, dts):

	aX = hg.GetX(matrix)
	aY = hg.GetY(matrix)
	aZ = hg.GetZ(matrix)

	# Cap, Pitch & Roll attitude:

	if aY.y > 0:
		y_dir = 1
	else:
		y_dir = -1

	horizontal_aZ = hg.Normalize(hg.Vec3(aZ.x, 0, aZ.z))
	horizontal_aX = hg.Cross(hg.Vec3.Up, horizontal_aZ) * y_dir
	horizontal_aY = hg.Cross(aZ, horizontal_aX)  # ! It's not an orthogonal repere !

	pitch_attitude = degrees(acos(max(-1, min(1, hg.Dot(horizontal_aZ, aZ)))))
	if aZ.y < 0: pitch_attitude *= -1

	roll_attitude = degrees(acos(max(-1, min(1, hg.Dot(horizontal_aX, aX)))))
	if aX.y < 0: roll_attitude *= -1

	heading = heading = degrees(acos(max(-1, min(1, hg.Dot(horizontal_aZ, hg.Vec3.Front)))))
	if horizontal_aZ.x < 0:
		heading = 360 - heading

	# axis speed:
	body_spd_x = hg.Dot(aX, physics_parameters["v_move"])
	body_spd_y = hg.Dot(aY, physics_parameters["v_move"])
	body_spd_z = hg.Dot(aZ, physics_parameters["v_move"])
	spdX = aX * body_spd_x
	spdY = aY * body_spd_y
	spdZ = aZ * body_spd_z

	frontal_speed = max(0.0, abs(body_spd_z))
	angle_of_attack = degrees(atan2(body_spd_y, max(1e-3, abs(body_spd_z))))
	sideslip_angle = degrees(atan2(body_spd_x, max(1e-3, abs(body_spd_z))))
	alpha_abs = abs(angle_of_attack)
	beta_abs = abs(sideslip_angle)

	# Thrust force:
	k = pow(physics_parameters["thrust_level"], 2) * physics_parameters["thrust_force"]
	# if self.post_combustion and self.thrust_level == 1:
	#    k += self.post_combustion_force
	F_thrust = aZ * k

	pos = hg.GetT(matrix)

	# Air density:
	air_density = compute_atmosphere_density(pos.y)
	# Dynamic pressure:
	q = hg.Vec3(pow(hg.Len(spdX), 2), pow(hg.Len(spdY), 2), pow(hg.Len(spdZ), 2)) * 0.5 * air_density
	total_speed = hg.Len(physics_parameters["v_move"])
	q_total = 0.5 * air_density * pow(total_speed, 2)

	alpha_peak_deg = physics_parameters.get("alpha_peak_deg", 18)
	alpha_limit_deg = physics_parameters.get("alpha_limit_deg", 28)
	alpha_departure_deg = physics_parameters.get("alpha_departure_deg", 36)
	deep_stall_alpha_deg = physics_parameters.get("deep_stall_alpha_deg", 45)
	post_stall_lift_factor = physics_parameters.get("post_stall_lift_factor", 0.45)
	high_alpha_drag_factor = physics_parameters.get("high_alpha_drag_factor", 2.0)
	alpha_protection = physics_parameters.get("alpha_protection", False)
	alpha_protection_gain = physics_parameters.get("alpha_protection_gain", 0.25)
	departure_speed_mps = physics_parameters.get("departure_speed_mps", 130)
	beta_departure_deg = physics_parameters.get("beta_departure_deg", 10)
	high_alpha_yaw_coupling = physics_parameters.get("high_alpha_yaw_coupling", 0.1)
	high_alpha_roll_coupling = physics_parameters.get("high_alpha_roll_coupling", 0.08)
	deep_stall_trim_gain = physics_parameters.get("deep_stall_trim_gain", 0.04)

	high_alpha = _smoothstep(alpha_peak_deg, alpha_limit_deg, alpha_abs)
	post_stall = _smoothstep(alpha_limit_deg, alpha_departure_deg, alpha_abs)
	deep_stall = _smoothstep(alpha_departure_deg, deep_stall_alpha_deg, alpha_abs)
	beta_departure = _smoothstep(beta_departure_deg * 0.5, beta_departure_deg, beta_abs)
	low_speed = 1.0 - _smoothstep(departure_speed_mps, departure_speed_mps * 1.6, frontal_speed)

	lift_scale = 1.0 + 0.2 * _smoothstep(0, alpha_peak_deg, alpha_abs)
	lift_scale = _lerp(lift_scale, 1.0, high_alpha * 0.5)
	lift_scale = _lerp(lift_scale, post_stall_lift_factor, post_stall)
	lift_scale *= (1.0 - 0.35 * deep_stall)
	lift_scale = max(0.15, lift_scale)

	drag_scale = 1.0 + 0.4 * high_alpha + high_alpha_drag_factor * post_stall + 1.5 * deep_stall + 0.25 * beta_departure
	flow_alignment = abs(body_spd_z) / max(total_speed, 1e-3)
	lift_q = _lerp(q.z, q_total, _smoothstep(0, alpha_peak_deg, alpha_abs))
	lift_q *= _lerp(1.0, max(0.35, flow_alignment), post_stall + 0.5 * deep_stall)

	# F Lift
	F_lift = aY * lift_q * physics_parameters["lift_force"] * lift_scale

	# Drag force:
	F_drag = (
		hg.Normalize(spdX) * q.x * physics_parameters["drag_coefficients"].x
		+ hg.Normalize(spdY) * q.y * physics_parameters["drag_coefficients"].y
		+ hg.Normalize(spdZ) * q.z * physics_parameters["drag_coefficients"].z
	) * drag_scale

	# Total

	physics_parameters["v_move"] += ((F_thrust + F_lift - F_drag) * physics_parameters["health_wreck_factor"] + F_gravity) * dts

	# Displacement:

	pos += physics_parameters["v_move"] * dts

	# Rotations:
	pitch_control_scale = max(0.15, 1.0 - 0.15 * high_alpha - 0.45 * post_stall - 0.6 * deep_stall)
	yaw_control_scale = max(0.15, 1.0 - 0.1 * high_alpha - 0.35 * post_stall - 0.45 * deep_stall)
	roll_control_scale = max(0.1, 1.0 - 0.15 * high_alpha - 0.5 * post_stall - 0.65 * deep_stall)

	F_pitch = physics_parameters["angular_levels"].x * q.z * physics_parameters["angular_frictions"].x * pitch_control_scale
	F_yaw = physics_parameters["angular_levels"].y * q.z * physics_parameters["angular_frictions"].y * yaw_control_scale
	F_roll = physics_parameters["angular_levels"].z * q.z * physics_parameters["angular_frictions"].z * roll_control_scale

	# Angular damping:
	gaussian = exp(-pow(frontal_speed * 3.6 * 3 / physics_parameters["speed_ceiling"], 2) / 2)

	# Angular speed:
	angular_speed = hg.Vec3(F_pitch, F_yaw, F_roll) * gaussian

	# Moment:
	pitch_m = aX * angular_speed.x
	yaw_m = aY * angular_speed.y
	roll_m = aZ * angular_speed.z

	# F-16-like high-alpha behavior: normal maneuvering is protected by an
	# AoA limiter, but once the aircraft is forced into a low-energy high-alpha
	# state, control effectiveness drops and departure tendencies appear.
	if alpha_protection and alpha_abs > alpha_limit_deg:
		alpha_guard = _smoothstep(alpha_limit_deg, alpha_departure_deg, alpha_abs)
		alpha_sign = 1.0 if angle_of_attack >= 0 else -1.0
		pitch_m += aX * (-alpha_sign * alpha_protection_gain * alpha_guard * q.z * physics_parameters["angular_frictions"].x)

	departure_factor = low_speed * max(post_stall, beta_departure * high_alpha)
	if departure_factor > 0:
		yaw_m += aY * (-physics_parameters["angular_levels"].z * high_alpha_yaw_coupling * q.z * physics_parameters["angular_frictions"].y * departure_factor)
		roll_m += aZ * ((sideslip_angle / max(beta_departure_deg, 1e-3)) * high_alpha_roll_coupling * q.z * physics_parameters["angular_frictions"].z * departure_factor)

	deep_stall_factor = low_speed * deep_stall
	if deep_stall_factor > 0:
		alpha_sign = 1.0 if angle_of_attack >= 0 else -1.0
		pitch_m += aX * (alpha_sign * deep_stall_trim_gain * q.z * physics_parameters["angular_frictions"].x * deep_stall_factor)

	# Easy steering:
	if physics_parameters["flag_easy_steering"]:
		# Keep the generic stabilizer effective in normal flight, but let it fade
		# once the aircraft is forced beyond the protected alpha regime.
		easy_steering_factor = _clamp(1.0 - 0.2 * high_alpha - 0.55 * post_stall - 0.85 * deep_stall, 0.05, 1.0)

		easy_yaw_angle = (1 - (hg.Dot(aX, horizontal_aX)))
		if hg.Dot(aZ, hg.Cross(aX, horizontal_aX)) < 0:
			easy_turn_m_yaw = horizontal_aY * -easy_yaw_angle
		else:
			easy_turn_m_yaw = horizontal_aY * easy_yaw_angle

		easy_roll_stab = hg.Cross(aY, horizontal_aY) * y_dir
		if y_dir < 0:
			easy_roll_stab = hg.Normalize(easy_roll_stab)
		else:
			n = hg.Len(easy_roll_stab)
			if n > 0.1:
				easy_roll_stab = hg.Normalize(easy_roll_stab)
				easy_roll_stab *= (1 - n) * n + n * pow(n, 0.125)

		zl = min(1, abs(physics_parameters["angular_levels"].z + physics_parameters["angular_levels"].x + physics_parameters["angular_levels"].y))
		roll_m += (easy_roll_stab * (1 - zl) + easy_turn_m_yaw) * q.z * physics_parameters["angular_frictions"].y * gaussian * easy_steering_factor

	# Moment:
	torque = yaw_m + roll_m + pitch_m
	axis_rot = hg.Normalize(torque)
	moment_speed = hg.Len(torque) * physics_parameters["health_wreck_factor"]

	# Return matrix:

	rot_mat = ms.MathsSupp.rotate_matrix(matrix, axis_rot, moment_speed * dts)
	mat = hg.TransformationMat4(pos, rot_mat)



	return mat, {
		"v_move": physics_parameters["v_move"],
		"pitch_attitude": pitch_attitude,
		"heading": heading,
		"roll_attitude": roll_attitude,
		"angle_of_attack": angle_of_attack,
		"sideslip_angle": sideslip_angle,
	}
