from movement import movement
movement_instance = movement()
position_filtered, velocity, time = movement_instance.loaddata('example_data')

print(position_filtered)
print(velocity)
print(time)