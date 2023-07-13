from loaddata import load_data
from decompose2D_file import decompose2D
position_filtered, velocity, time = load_data('data\subject08day1pre')
numsubmovements = 4
xrng = (-20, 20)
yrng = (-10, 10)
best_error, final_parms = decompose2D(time=time[0],vel=velocity[0],
                                      nSubmovements=numsubmovements,xrng=xrng,yrng=yrng)
print(best_error)
print(final_parms)
