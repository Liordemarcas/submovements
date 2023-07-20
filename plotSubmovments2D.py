import numpy as np
from minimumJerkVelocity2D import minimumJerkVelocity2D
import matplotlib.pyplot as plt


def plotSubmovements2D(parameters,x0,y0,plottype = 1,t = None):

    if int(len(parameters))%4 != 0:
        raise ValueError('The parameters vector must have a length that is a multiple of 4')

    numsubmovements = int(len(parameters)/4)


    t0 = parameters[:, 0]
    D = parameters[:, 1]
    Ax = parameters[:, 2]
    Ay = parameters[:, 3]

    # t0 = parameters[0:len(parameters)-4:4]
    # D = parameters[1:len(parameters)-3:4]
    # Ax = parameters[2:len(parameters)-2:4]
    # Ay = parameters[3:len(parameters)-2:4]

    order = np.argsort(t0)
    t0 = t0[order]
    D = D[order]
    Ax = Ax[order]
    Ay = Ay[order]

    x0 = np.concatenate((x0,x0 + np.cumsum(Ax[0:-1])))
    y0 = np.concatenate((y0,y0 + np.cumsum(Ay[0:-1])))
 
    tf = t0 + D

    if t is None:
        t = np.linspace(min(t0),max(tf),num=100)
        print (t)
    vx = np.zeros((numsubmovements,t.size))
    vy = np.zeros((numsubmovements,t.size))

    for isub in range(numsubmovements):

        vx[isub,:], vy[isub,:], _ = minimumJerkVelocity2D(t0[isub],D[isub], Ax [isub], Ay[isub],t)
    
    print('vx')
    print(vx)
    print('vy')
    print(vy)

    # fig, axs = plt.subplots(1, 1)
    # axs.plot(t,vx.transpose(),'b',label = "vx" )
    # axs.plot(t,vy.transpose(),'r', label = "vy")

    # plt.show()
    





# if any(plottype==1:2)
#     h(1:2:numSubmovements*2-1) = plot(t,vx,'b');
#     hold on;
#     h(2:2:numSubmovements*2) = plot(t,vy,'r');
#     legend(h(1:2),'Submovements v_x','Submovements v_y');
#     xlabel('time');
#     ylabel('velocity');
# end










# for s = 1:numSubmovements    
#     [vx(s,:),vy(s,:)] = minimumJerkVelocity2D(t0(s),D(s),Ax(s),Ay(s),t);
#     [x(s,:),y(s,:)] = minimumJerkPosition2D(t0(s),D(s),Ax(s),Ay(s),x0(s),y0(s),t);
# end

    # print (tf)

    # print(t0)
    # print (D)
    # print(Ax)
    # print(Ay)

    return t0, D, Ax, Ay