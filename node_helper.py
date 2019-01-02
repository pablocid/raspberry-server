import sys, getopt, socket, time
def main(argv):
    instruction = ''
    inputname = ''
    try:
        opts, args = getopt.getopt(argv,"hi:n:",["instruction=", 'photoname='])
    except getopt.GetoptError:
        print('-i <inputinstruction> -n <photoname>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('-i <inputinstruction>')
            sys.exit()
        elif opt in ("-i", "--instruction"):
            instruction = arg
        elif opt in ("-n", "--photoname"):
            inputname = arg

    try:
        s = socket.socket()
        s.settimeout(3)
        s.connect(('localhost', 8008))
        if len(inputname)>0:
            s2 = socket.socket()
            s2.settimeout(3)
            s2.connect(('localhost', 8009))
    except:
        raise ConnectionError
    try:
        s.send(instruction.encode('utf-8'))
        a=s.recv(1024).decode('utf-8')
        if a == 'done':
            if len(inputname) > 0:
                try:
                    s2.send(inputname.encode('utf-8'))
                    b = s2.recv(1024).decode('utf-8')
                    print(b, end='')
                    sys.exit()
                except:
                    print('time_out', end='')
                    sys.exit()
            else:
                print(a, end='')
                sys.exit()
    except:
        print('time_out', end='')
        sys.exit()



if __name__ == "__main__":
   main(sys.argv[1:])
