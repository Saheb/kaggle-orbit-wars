env4 = make("orbit_wars", debug=True)
env4.run([nearest_planet_sniper, nearest_planet_sniper, nearest_planet_sniper, nearest_planet_sniper])

final = env4.steps[-1]
for i, s in enumerate(final):
    print(f"Player {i}: reward={s.reward}, status={s.status}")

env4.render(mode="ipython", width=800, height=600)

agent = nearest_planet_sniper
