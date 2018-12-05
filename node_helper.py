import sys, getopt, socket, time
def main(argv):
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:",["instruction=",])
    except getopt.GetoptError:
        print('-i <inputinstruction>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('-i <inputinstruction>')
            sys.exit()
        elif opt in ("-i", "--instruction"):
            inputfile = arg
    s = socket.socket()
    s.settimeout(1)
    try:
        s.connect(('localhost', 8008))
    except:
        raise ConnectionError
    try:
        s.send(inputfile.encode('utf-8'))
        a=s.recv(1024).decode('utf-8')
    except:
        raise ConnectionError('time out')
    #time.sleep(0.5)
    #print(a)
    return a
    sys.exit()

if __name__ == "__main__":
   main(sys.argv[1:])
