
# Step 1: load the data
from loaddata import load_data
position_filtered, velocity, time = load_data('data/subject08day1pre')
# Step 1 is finished. We're able to load all CSVs from a directory



# Step 2: Plot the data to become familiar with it

from plot import plot_position, plot_velocity
plot_position(position_filtered, time)
plot_velocity(velocity, time)


#Step 3: Decompose a single trial into 4 submovements
####from decompose_file load decompose2d
