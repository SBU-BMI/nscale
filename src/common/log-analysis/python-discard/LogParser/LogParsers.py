'''
Created on Feb 19, 2013

List of parsers.  Each one is specific to the version of log.

@author: tcpan
'''
import re, pprint


class LogParserBase():
    '''
    this is the base class. all other parsers inherit from this
    supports streaming so only parse one line at a time.
    '''
    pp = pprint.PrettyPrinter(indent=4)
    experiment = {}

    def __init__(self,filename):
        '''
        Constructor
        '''
        self.parseFilename(filename)
        
    def parseFilename(self, filename):
        '''
        parsing filename
        '''
        # initialize
        self.experiment['filename']    = filename  
        self.experiment['type']        = "synth" if "syntest" in filename else "TCGA GBM"     
        self.experiment['layout']      = "separate"
        self.experiment['nMaster']     = 0 if "push" in filename else 1
        self.experiment['nCompute']    = -1       
        self.experiment['nRead']       = -1       
        self.experiment['nWrite']      = -1       
        self.experiment['transport']   = "opencv"    
        self.experiment['bufferSize']  = 1     
        self.experiment['ioGroupSize'] = -1       
        self.experiment['dataSize']    = 4096        
        self.experiment['blocking']    = True         
        self.experiment['compression'] = False         
        self.experiment['ost']         = True if ".ost." in filename else False 
        
        # parse the filename
        if "kfs" in filename:
            self.experiment['sys'] = "kfs"           
        elif "kids" in filename:
            self.experiment['sys'] = "kids"           
        elif "jaguar" in filename:
        	self.experiment['sys'] = "jaguar"
        elif "titan" in filename:
            self.experiment['sys'] = "titan"
        else:
            self.experiment['sys'] = "keeneland"

        if .+\/synthetic\.datasizes\.p([0-9]+)\.push\.([^\.]+)\.([0-9]+)$

        # submit the parameter to the database, get back the unique experiment id.

        
                                                     
    def parseLine(self, line, handler):
        '''
        parses one line of the log file.
        takes a handler to upload data to database
        '''
        print "base parser"
        
    def submitToDB(self, data, handler):
        '''
        submit to database
        '''
        self.pp.pprint(data);
        
        
class LogParserV2_1(LogParserBase):
    '''
    this is log parser for v 2.1.
    supports streaming so only parse one line at a time.
    '''
        
    def parseLine(self, line, handler):
        '''
        parses one line of the log file.
        takes a handler to upload data to database
        '''
        # check for empty line
        l = line.strip();
        if len(l) == 0:
            return
        
        # get the process id
        event = {}
        p=re.compile('pid,(\d+),hostName,([^,]+),group,(\d+),sessionName,([^,]+),')
        m=p.match(l)
        if m:
            event['pid'] = m.group(1)
            event['hostname'] = m.group(2)
            event['group'] = m.group(3)
            event['sessionname'] = m.group(4)
        else:
            print "no match: {0}".format(l)
            return
        
        # now extract the events
        p = re.compile('([^,]+),(\d+),(\d+),(\d+),(\d*),')
        it = p.finditer(l)
        for m in it:
            event['name'] = m.group(1)
            event['type'] = m.group(2)
            event['start'] = m.group(3)
            event['stop'] = m.group(4)
            event['attr'] = m.group(5)
            
            self.submitToDB(event, handler)
            

class LogParserV2(LogParserBase):
    '''
    this is log parser for v 2.1.
    supports streaming so only parse one line at a time.
    '''

        
    def parseLine(self, line, handler):
        '''
        parses one line of the log file.
        takes a handler to upload data to database
        '''
        # check for empty line
        l = line.strip();
        if len(l) == 0:
            return
        
        event = {}

        # get the process id
        p=re.compile('pid,(\d+),hostName,([^,]+),sessionName,([^,]+),')
        m=p.match(l)
        if m:
            event['pid'] = m.group(1)
            event['hostname'] = m.group(2)
            event['sessionname'] = m.group(3)
        else:
            print "no match: {0}".format(l)
            return
        
        # now extract the events
        p = re.compile('([^,]+),(\d+),(\d+),(\d+),(\d*),')
        it = p.finditer(l)
        for m in it:
            event['name'] = m.group(1)
            event['type'] = m.group(2)
            event['start'] = m.group(3)
            event['stop'] = m.group(4)
            event['attr'] = m.group(5)
            
            self.submitToDB(event, handler)
        

class LogParserV1(LogParserBase):
    '''
    this is log parser for v 2.1.
    supports streaming so only parse one line at a time.
    '''

        
    def parseLine(self, line, handler):
        '''
        parses one line of the log file.
        takes a handler to upload data to database
        '''
        # check for empty line
        l = line.strip();
        if len(l) == 0:
            return

        event = {}
        
        # get the process id
        p=re.compile('pid,(\d+),hostName,([^,]+),sessionName,([^,]+),')
        m=p.match(l)
        if m:
            event['pid'] = m.group(1)
            event['hostname'] = m.group(2)
            event['sessionname'] = m.group(3)
        else:
            print "no match: {0}".format(l)
            return
        
        # now extract the events
        p = re.compile('([^,]+),(\d+),(\d+),(\d+),')
        it = p.finditer(l)
        for m in it:
            event['name'] = m.group(1)
            event['type'] = m.group(2)
            event['start'] = m.group(3)
            event['stop'] = m.group(4)
            
            self.submitToDB(event, handler)
        
    