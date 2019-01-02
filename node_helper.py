import sys, getopt, socket, time

errors_dict_google={'duplicated_name':'201', 'added_online':'200', 'added_offline':'202'}

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
    s = socket.socket()
    s.settimeout(3)
    if len(inputname)>0:
        s2 = socket.socket()
        s2.settimeout(3)
    try:
        s.connect(('localhost', 8008))
        if len(inputname)>0:
            s2.connect(('localhost', 8009))
    except:
        raise ConnectionError
    try:
        s.send(instruction.encode('utf-8'))
        a=s.recv(1024).decode('utf-8')
        if 'done' in a:
            if len(inputname) > 0:
                s2.send(inputname.encode('utf-8'))
                b = s2.recv(1024).decode('utf-8')
    except:
        print('time_out', end='')
        sys.exit()
    if len(inputname) > 0:
        print(errors_dict_google[b], end='')
        sys.exit()
    else:
        print(a, end='')
        sys.exit()

if __name__ == "__main__":
   main(sys.argv[1:])
