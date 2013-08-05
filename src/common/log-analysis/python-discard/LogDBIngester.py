#!/usr/local/bin/python

# check python version
import sys
ver_info = sys.version_info

# parse commandlines
if ver_info[0] < 3 and ver_info[1] < 7:
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filename", help="input log file", metavar="LOG_FILE")
#    parser.add_option("-d", "--directory", dest="dirname", help="input directory with log files", metavar="LOG_DIR")
    parser.add_option("-t", "--dbtype", dest="dbtype", help="database type", default="mongodb", metavar="DB_TYPE")
    (options, args) = parser.parse_args();

else:
    import argparse
    parser = argparse.ArgumentParser(description="Log to database ingester")
    parser.add_argument("-f, --file", dest="filename", help="input log file", metavar="LOG_FILE")
#    parser.add_argument("-d, --directory", dest="dirname", help="input directory with log files", metavar="LOG_DIR")
    parser.add_argument("-t, --dbtype", dest="dbtype", help="database type", default="mongodb", metavar="DB_TYPE")
    options = parser.parse_args()


print "file {0} ".format(options.filename)
#    print "dirname {0} ".format(options.dirname)
print "dbtype {0}".format(options.dbtype)


if options.dbtype == "mongodb":
    from DBDriver.MongoDBDriver import MongoDBDriver
    dbingester = MongoDBDriver();
elif options.dbtype == "cassandra":
    from DBDriver.CassandraDBDriver import CassandraDBDriver
    dbingester = CassandraDBDriver();
else:
    print "ERROR: unsupported db type {0}".format(options.dbtype);
    sys.exit(2);

import re
# open the file and iterate
with open(options.filename) as f:
    # read the first line
    line = f.readline()
    if re.match("v2.1", line):
        from LogParser.LogParsers import LogParserV2_1
        lparser = LogParserV2_1(options.filename)
    elif re.match("v2", line):
        from LogParser.LogParsers import LogParserV2
        lparser = LogParserV2_1(options.filename)        
    else:
        print "UNSUPPORTED LOG VERSION: {0}".format(line)
        sys.exit(1)
    
    
    for line in f:
        lparser.parseLine(line, dbingester)
        
