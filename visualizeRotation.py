#!/usr/bin/env pvpython
'''
Visualize some vectors in ParaView.  Requires ParaView with pvpython.
'''

#from __future__ import print_function

from paraview.simple import *

import select
import socket
import struct
import numpy as np

from misc_utils import get_last_packet



cone1 = Cone(Radius=0.1, Center=[0,0,0])
Show()
cone2 = Cone(Radius=0.1, Center=[0,0,0])
Show()
cone3 = Cone(Radius=0.1, Center=[0,0,0])
Show()
Render()
ResetCamera()
sources = GetSources()
renderView1 = GetActiveViewOrCreate('RenderView')


cone1Display = GetDisplayProperties(cone1, view=renderView1)
cone2Display = GetDisplayProperties(cone2, view=renderView1)
cone3Display = GetDisplayProperties(cone3, view=renderView1)
cone1Display.DiffuseColor = [1.0, 0, 0]
cone2Display.DiffuseColor = [1.0, 1.0, 0]
cone3Display.DiffuseColor = [0, 0, 1.0]


print("cone1:", cone1)
print("sources:", sources)
print("renderView1:", renderView1)

DATA_PORT = 12311
data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
data_sock.bind(("127.0.0.1", DATA_PORT))


def updateRotation(R):
    print(R)

    cone1.Direction = np.dot(R, np.array([1,0,0]))
    cone2.Direction = np.dot(R, np.array([0,1,0]))
    cone3.Direction = np.dot(R, np.array([0,0,1]))
    renderView1.Update()
    RenderAllViews()

    return
    x1, y1, z1, x2, y2, z2, x3, y3, z3 = struct.unpack('!fffffffff', data)
    cone1.Direction = [x1, y1, z1]
    cone2.Direction = [x2, y2, z2]
    cone3.Direction = [x3, y3, z3]
    renderView1.Update()
    RenderAllViews()


while True:
    inputs, outputs, errors = select.select([data_sock], [], [])
    for oneInput in inputs:
        if oneInput == data_sock:
            pkt, addr = get_last_packet(data_sock)
            if pkt is not None:
                if len(pkt) == 8*9:
                    R = np.frombuffer(pkt)
                    R = R.reshape((3,3))
                    updateRotation(R)
                else:
                    print('failed to parse UDP packet')
                    print(len(pkt))


