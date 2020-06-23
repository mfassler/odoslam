
def get_last_packet(sock, bufsize=65536):
    '''Empty out the UDP recv buffer and return only the final packet
    (in case the GUI is slower than the data flow)
    '''
    sock.setblocking(0)
    data = None
    addr = None
    cont = True
    while cont:
        try:
            tmpData, addr = sock.recvfrom(bufsize)
        except Exception as ee:
            #print(ee)
            cont=False
        else:
            if tmpData:
                if data is not None:
                    pass
                    #print('throwing away a packet (GUI is too slow)')
                data = tmpData
            else:
                cont=False
    sock.setblocking(1)
    return data, addr

