# Strike Eagle - Missiles & Aircraft Reference

## Aircraft in the Simulator

### F-16 Fighting Falcon (Agent)

- **Role:** Lightweight multirole fighter
- **Origin:** United States (General Dynamics / Lockheed Martin)
- **Sim ID:** `ally_1`
- **Characteristics:** Single-engine, highly agile, excellent for evasive
  maneuvering. The "sports car" of fighter jets.
- **Weapons:** 12 missile slots (AIM Sidewinder, Karaoke, CFT variants)

### Eurofighter Typhoon (Enemy)

- **Role:** Twin-engine multirole fighter
- **Origin:** European consortium (UK, Germany, Italy, Spain)
- **Sim ID:** `ennemy_1`
- **Characteristics:** Twin-engine, very capable, carries long-range missiles.
  Used as the missile launch platform in our scenario.
- **Weapons:** 6 missile slots (Meteor, MICA)

### Also Available

- **Rafale** — French twin-engine fighter (similar to Eurofighter)
- **TFX (TAI Kaan)** — Turkish 5th-gen stealth fighter (in development IRL)
- **F-14 Tomcat** — US Navy swing-wing fighter (Top Gun fame)
- **Miuss** — Russian concept UCAV

## Missiles in the Simulator

### Air-to-Air Missiles (AAM)

**AIM Sidewinder** (`ally_1AIM_SL0/SL1`)

- Short-range, infrared (heat-seeking)
- Tracks target engine heat signature
- Range: ~20 km
- The classic dogfight missile

**MICA** (`ennemy_1Mica1-4`)

- French medium-range missile
- Can be IR (heat) or radar guided
- Range: ~80 km
- Used on Rafale and Eurofighter

**Meteor** (`ennemy_1Meteor0/5`)

- European beyond-visual-range AAM
- Ramjet-powered (breathes air = sustained speed)
- Range: 100+ km
- One of the most capable AAMs in the world
- **This is what the enemy fires at our agent**

**Karaoke** (`ally_1Karaoke2-5/8-11`)

- Sim-specific medium-range missile

### Surface-to-Air Missiles (SAM)

**S-400** (`Ennemy_Missile_launcher_1S4000-4003`)

- Russian long-range SAM system
- Range: up to 400 km, ceiling 30 km
- One of the most feared air defense systems in the world
- Already present in the sandbox scene as a ground launcher
- Comparable to the US Patriot system

## Evasion Concepts

### Maneuvers

- **Notching** — Turning perpendicular to a radar-guided missile to blend
  into ground clutter. The missile's radar can't distinguish the aircraft
  from terrain returns.

- **Cranking** — Flying at the edge of the missile's tracking cone to
  maximize the distance it must travel while keeping situational awareness.

- **Break Turn** — Last-resort hard turn into the missile to force an
  overshoot. Missiles have limited G-force for turning at high speed.

- **Going Cold** — Turning directly away from the missile to maximize
  the closure distance. Effective against long-range missiles that might
  run out of fuel.

- **Altitude Changes** — Diving forces the missile to follow through
  denser air (more drag). Climbing then diving can cause tracking issues.

### What Makes Missiles Hard to Dodge

Missiles fly at Mach 3-4+ but have limited fuel and turning ability.
The closer a missile gets, the harder it is to evade because it has more
energy for course corrections. The best evasion strategy combines:

1. **Distance** — Make the missile fly further (more fuel burned)
2. **Energy bleeding** — Force the missile to turn (bleeds speed)
3. **Timing** — Maneuver at the right moment (too early = missile corrects,
   too late = can't escape)

This is exactly what we're training the RL agent to discover.
