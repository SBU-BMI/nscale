'''
Created on Feb 19, 2013

@author: tcpan
'''

from DBDriver.DBDriverBase import DBDriverBase; 


class CassandraDBDriver(DBDriverBase):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        print "initialized CassandraDBDriver"
        
    def create(self, entry):
        '''
        upload data to database
        '''
        print "cass create"
        
    def retrieve(self):
        '''
        blah
        '''
    def update(self):
        '''
        blah
        '''

    def delete(self):
        '''
        delete data from database
        '''
        