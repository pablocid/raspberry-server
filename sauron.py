import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
import datetime
import socket
import threading

class Entry_control():
    def __init__(self):
        self.ipid='71'
        self.entry=''
        self.breaker=False
        try:
            self.scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
            self.creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', self.scope)
            self.client=None
            self.sheet=None
            self.connect_gspreads()
            self.try_sync()
        except FileNotFoundError:
            print('no json credential found')
        except:
            print("Can't access web service")
        t1 = threading.Thread(target=self.server_call, )
        t1.start()
        print('google is watching you')

    def server_call(self):
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('localhost', 8009))
        s.listen(1)
        while True:
            if self.breaker:
                s.close()
                break
            c, addr = s.accept()
            a = c.recv(1024).decode('utf-8')
            if self.add_new(a):
                c.send('_'.join(['added', self.gstate]).encode('utf-8'))
            else:
                c.send('duplicated_name'.encode('utf-8'))

    def connect_gspreads(self):
        self.client = gspread.authorize(self.creds)
        self.sheet = self.client.open(" Laboratorio 2018-2019").worksheet("Fotos")

    def add_new(self, name):
        if not self.query_local(name):
            try:
                self.try_sync()
                if len(self.sheet.findall(name))==0:
                    self.sheet.append_row([name, self.ipid, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")])
                    self.gstate='online'
                    return True
                else:
                    return False
            except:
                self.write_to_tsv([name, self.ipid, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), 'no_sync'])
                self.gstate = 'offline'
                return True
        else:
            return False

    def load_local(self):
        temp=[]
        if 'local_spreadsheet.tsv' not in os.listdir('./'):
            file = open('./local_spreadsheet.tsv', 'w')
            file.close()
            return temp
        file = open('./local_spreadsheet.tsv', 'r')
        text=file.read().split('\n')
        file.close()
        for n in text:
            if len(n)<=2:
                continue
            temp.append(n.split('\t'))
        return temp

    def write_to_tsv(self, entry):
        file = open('./local_spreadsheet.tsv', 'a')
        file.write('\n'+'\t'.join(entry))
        file.close()

    def query_local(self, name):
        file = open('./local_spreadsheet.tsv', 'r')
        if name in file.read():
            return True
        else:
            return False

    def try_sync(self):
        temp=self.load_local()
        file = open('./local_spreadsheet.tsv', 'w')
        file.close()
        for n in range(len(temp)):
            if temp[n][3]=='no_sync':
                try:
                    self.sheet.append_row(temp[n][:3])
                except:
                    self.write_to_tsv(temp[n])

class Entry_control_ondemand():
    def __init__(self, entry):
        self.ipid='71'
        self.entry=entry
        self.breaker=False
        try:
            self.scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
            self.creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', self.scope)
            self.client=None
            self.sheet=None
            self.connect_gspreads()
            self.try_sync()
        except FileNotFoundError:
            print('no json credential found')
        except:
            print("Can't access web service")
        #return self.ondemand(name=self.entry)

    def ondemand(self):
        if self.add_new(self.entry):
            return '_'.join(['added', self.gstate])
        else:
            return 'duplicated_name'


    def server_call(self):
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('localhost', 8009))
        s.listen(1)
        while True:
            if self.breaker:
                s.close()
                break
            c, addr = s.accept()
            a = c.recv(1024).decode('utf-8')
            if self.add_new(a):
                c.send('_'.join(['added', self.gstate]).encode('utf-8'))
            else:
                c.send('duplicated_name'.encode('utf-8'))

    def connect_gspreads(self):
        self.client = gspread.authorize(self.creds)
        self.sheet = self.client.open(" Laboratorio 2018-2019").worksheet("Fotos")

    def add_new(self, name):
        if not self.query_local(name):
            try:
                self.try_sync()
                if len(self.sheet.findall(name))==0:
                    self.sheet.append_row([name, self.ipid, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")])
                    self.gstate='online'
                    return True
                else:
                    return False
            except:
                self.write_to_tsv([name, self.ipid, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), 'no_sync'])
                self.gstate = 'offline'
                return True
        else:
            return False

    def load_local(self):
        temp=[]
        if 'local_spreadsheet.tsv' not in os.listdir('./'):
            file = open('./local_spreadsheet.tsv', 'w')
            file.close()
            return temp
        file = open('./local_spreadsheet.tsv', 'r')
        text=file.read().split('\n')
        file.close()
        for n in text:
            if len(n)<=2:
                continue
            temp.append(n.split('\t'))
        return temp

    def write_to_tsv(self, entry):
        file = open('./local_spreadsheet.tsv', 'a')
        file.write('\n'+'\t'.join(entry))
        file.close()

    def query_local(self, name):
        file = open('./local_spreadsheet.tsv', 'r')
        if name in file.read():
            return True
        else:
            return False

    def try_sync(self):
        temp=self.load_local()
        file = open('./local_spreadsheet.tsv', 'w')
        file.close()
        for n in range(len(temp)):
            if temp[n][3]=='no_sync':
                try:
                    self.sheet.append_row(temp[n][:3])
                except:
                    self.write_to_tsv(temp[n])

if __name__ == '__main__':
    Entry_control()