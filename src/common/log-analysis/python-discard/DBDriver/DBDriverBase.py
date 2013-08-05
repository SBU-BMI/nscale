'''
Created on Feb 19, 2013

@author: tcpan
'''


class DBDriverBase:
    '''
    all database access layers inherit from this
    '''


    def __init__(self):
        '''
        Constructor
        '''
        print "initialized DBDriverBase"

    def create(self, entry):
        '''
        upload data to database
        '''
        print "db create"
        
    def retrieve(self):
        '''
        upload data to database
        '''

    def update(self):
        '''
        upload data to database
        '''

    
    def delete(self):
        '''
        delete data from database
        '''
        