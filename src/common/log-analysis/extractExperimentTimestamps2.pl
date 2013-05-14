#!/usr/bin/perl

# TODO:  reformulate to extract block representing a single experiment, and then process it.



use strict;
use warnings;

use File::stat;
use Time::localtime;
use Time::Piece;

use local::lib;
use Parallel::ForkManager 0.7.6;

use lib('.');
use cci_common_filenameparser2;

use File::Find;

# get the list of directories
my($rootdir) = '/home/tcpan/PhD/path/Data/adios';
my(@dirnames);
# exclude PreIPROBE itself and rootdir itself.
find(sub { push(@dirnames, $File::Find::name) if (-d $_ && $_ !~ /^PreIPROBE$/ && $File::Find::name !~ /^$rootdir$/); }, $rootdir);
print "DIR TO PROCESS: $_\n" foreach (@dirnames);


# multithreaded run
my($pm) = Parallel::ForkManager->new(8);
# data structure retrieval and handling
$pm -> run_on_finish ( # called BEFORE the first call to start()
  sub {
    my ($pid, $exit_code, $ident, $exit_signal, $core_dump, $data_structure_reference) = @_;

    # retrieve data structure from child
    if (defined($data_structure_reference)) {  # children are not forced to send anything
      my $string = ${$data_structure_reference};  # child passed a string reference
      print "$string";
    } else {  # problems occuring during storage or retrieval will throw a warning
      #print qq|No message received from child process $pid!\n|;
    }
  }
);

#  $pm->run_on_start(
#    sub { my ($pid,$ident)=@_;
#      print "** $ident started, pid: $pid\n";
#    }
#  );


# create the header
my($header1) = "directory,run,experiment,start,end,hasfile,expfiletime";
my($header) = "type,layout,sys,nProcs,nMaster,nCompute,nIO,fileCount,transport,ioGroupSize,bufferSize,dataSize,blocking,compression,ost";
# node types are (m/assign), io, seg, w(=io,seg)
my(@colNames) = split(",", $header);
print "$header1";
print ",f.$_,x.$_" foreach (@colNames);
print "\n";


# check the output console log buffer for errors
sub checkForError($$$) {
	
	my($strbuf, $timestr, $runEnd) = @_;
	my($endTime) = $timestr;
	
	# check if any error occured
	if ( $strbuf =~ /MPI(CH2)? ERROR/i) {
		$endTime = "MPI ERROR";
	} elsif ($strbuf =~ /exit signal (Aborted|Terminated|Segmentation fault)/) {
		$endTime = $1;
	} elsif ($strbuf =~ /Invalid argument/) {
		$endTime = "INVALID ARG";
	} elsif ($strbuf =~ /Other I\/O error/) {
		$endTime = "IO ERROR";
	} elsif ($strbuf =~ /exceeded/i) {
		$endTime = "> $runEnd";
	} elsif ($strbuf =~ /\n([^\n]*fail|error[^\n]*)\n/i) {
		$endTime = "OTHER FAILURE";
		print STDERR "ERROR: $1\n"; 
	}
	
	return $endTime;	
}

# reformatting the time string to something excel can understand
sub reformatTimeString($) {
	
	my($timestr) = @_;
	
	my($t);
	if ($timestr =~ /\w{3} \w{3} \s?\d+ \d{2}:\d{2}:\d{2} EDT 201[23]/) {
		$t = Time::Piece->strptime($timestr,
    	                            "%a %b %e %T EDT %Y");
	} elsif ($timestr =~ /\w{3} \w{3} \s?\d+ \d{2}:\d{2}:\d{2} EST 201[23]/) {
		$t = Time::Piece->strptime($timestr,
    	                            "%a %b %e %T EST %Y");
	} elsif ($timestr =~ /\w{3} \w{3} \s?\d+ \d{2}:\d{2}:\d{2} 201[23]/) {
		$t = Time::Piece->strptime($timestr,
    	                            "%a %b %e %T %Y");		
	} else {
		print STDERR "ERROR: date format not supported: $timestr\n";
		return "";
	}
	return $t->date . " " . $t->time;
	
}


# iterate through the directoires
foreach my $dirname (@dirnames) {
	
	# start a new task
	my $pid = $pm->start($dirname) and next;  # master will go on next because pm->start returns non-zero.	
	
	## child task will start running here
	
	# get the subdir name within the rootdir
	$dirname =~ /$rootdir\/(.*)/;
	my($dn) = $1;

	#print STDERR "$rootdir, $dirname, $dn\n";
	
	# find all the experiments in the directory that we have log for.  i.e. csv files
	my(%expdates);
	my(%expfparams);
	my(%expused);	
	my(@csvs) = <$dirname/*.csv>;
	foreach my $csv (@csvs) {
		# extract the experiment filename and actual experiment name
		$csv =~ /^(.*?)([\-]na[\-]POSIX|[\-]na[\-]NULL|[\-]NULL|[\-]POSIX|[\-]MPI_AMR|[\-]MPI_LUSTRE|[\-]MPI)?\.csv$/;
		my($expname) = $1;
		
		# get the parameters from the filename, and the file's timestamp
		$expfparams{$expname} = parseFileName($expname);
		$expdates{$expname} = reformatTimeString(ctime(stat($csv)->mtime));
		$expused{$expname} = 0;
#		print "$expfn, $expname: $expdates{$expname}\n" if $dirname =~ /tcga.p2048/;
	}

	my($outputstr) = "";

	# get the console logs
	my(@filenames) = (<$dirname/*.o2*>, <$dirname/*.o1*>);
	foreach my $filename (@filenames) {
		next if (stat($filename)->size <= 0);  # skip any file that are 0-byte in size
	
		# get the finish time stamp of the experiment set.  - this is a GROSS APPROXIMATION of END
		my($runEnd) = reformatTimeString(ctime(stat($filename)->mtime));

		# read the console log
		local *FH;
		open(FH, $filename) or die("ERROR: unable to open $filename!\n");
		my $data;
	 	{
		  -f FH and sysread FH, $data, -s FH;
		}
		close(FH);
	
		# get the first timestamp in the console log.  this is a GROSS APPROXIMATION of START
		$data =~ /^(.* E[DS]T 201[23])$/m;  # per line.
		my($runStart) = reformatTimeString($1) if (defined($1));
		my($count) = 0;
		my($currPos) = -1;
		
		my($startTime) = $runStart;
		my($endTime) = undef;
		my($experiment) = undef;
		my($expFileTime) = undef;
		my($fparams);
		my(%xparams);
	
		$filename =~ /$dirname\/([^\/]*)/;	
		my($runname) = $1;
	#	print $runname,"\n";
	
		#iterating over the data with global reg exp match does not work with \G even though
		#documentation suggests that this is the right way to do it. 
		pos($data) = undef;
		my($len);
		
		if ($filename =~ /titan/) {

			while ( $data =~ /([^\n]*? E[DS]T 201[23])\n(mpirun|aprun) (.*?) (.*?\.exe) ([^\n]*)\n(.*?)?(Recognized Options:\n)?(.*?)?dir to create:/gm ) {
				# write out the previous experiment
				if (defined($experiment)) {
					$len = pos($data) - $currPos;
					$endTime = checkForError(substr($data, $currPos, $len), reformatTimeSTring($1), $runEnd);
					
					# output previously found experiment
					$outputstr .= "$dn,$runname,$experiment,$startTime,";
					if (defined($endTime)) {
						$outputstr .= "$endTime";		
					} else {
						$outputstr .= "UNDEFINED";
					}
					if (defined($expFileTime)) {
						$outputstr .= ",$expFileTime";			
					} else {
						$outputstr .= ",NA";
					}
					
					# output the captured parameters
					foreach my $colN (@colNames) {
						if (defined($fparams) && defined($fparams->{$colN})) {
							$outputstr .= ",$fparams->{$colN}";
						} else {
							$outputstr .= ",";
						}
						if (defined($xparams{$colN})) {
							$outputstr .= ",$xparams{$colN}";
						} else {
							$outputstr .= ",";
						}
					}
					$outputstr .= "\n";
				} 
					
					
					
				$currPos = pos($data);
				
				my($paramblock1) = $5;
				my($paramblock2) = $8;
				my($marker) = $2;
				my($executable) = $4;
				# now set new experiment data
				$startTime = reformatTimeString($1);
				$endTime = undef;
				%xparams = ();
				$expFileTime = undef;
				$count++;
				my($paramblock);
				
				if ($executable =~ /nu-segment-scio\.exe/) {
					$paramblock = $paramblock1;
					
					if ($paramblock =~ /(\d+) ([^ ]+) [cg]pu/ ) {
						$xparams{type} = "TCGA";
						$xparams{layout} = "baseline";
						$xparams{sys} = "?";
						$xparams{nProcs} = "?";
						$xparams{nMaster} = 1;
						$xparams{nCompute} = "all";
						$xparams{nIO} = "all";
						$xparams{fileCount} = $1;
						$xparams{transport} = "opencv";
						$xparams{ioGroupSize} = 1;
						$xparams{bufferSize} = 1;
						$xparams{dataSize} = 4096;
						$xparams{blocking} = "n";
						$xparams{compression} = "n";
						$xparams{ost} = "n";
						
					} else {
						print STDERR "3. $filename: $executable: $paramblock\n";
					}
					
				} elsif ($executable =~ /nu-segment-scio-adios\.exe/) {
					$paramblock = $paramblock1;
					
					if ($paramblock =~ /([^ ]+) (\d+) (\d+) [cg]pu ([\-]?\d+) \d+/ ) {
						$xparams{type} = "TCGA";
						$xparams{layout} = "coloc";
						$xparams{sys} = "?";
						$xparams{nProcs} = "?";
						$xparams{nMaster} = 1;
						$xparams{nCompute} = "all";
						$xparams{nIO} = "all";
						$xparams{fileCount} = $2;
						$xparams{transport} = $1;
						$xparams{ioGroupSize} = $4;
						$xparams{bufferSize} = $3;
						$xparams{dataSize} = 4096;
						$xparams{blocking} = "n";
						$xparams{compression} = "n";
						$xparams{ost} = "n";
						
					} else {
						print STDERR "4. $filename: $executable: $paramblock\n";
					}
				} elsif ($executable =~ /SegmentNucleiReader\.exe/) {
					$paramblock = $paramblock2;
					
					
					
				}
				$experiment = $paramblock =~ /log:[ ]+([^ \n]+)/ ? $1 : undef;
				$fparams = parseFileName($1);
				if (!defined($fparams)) {
					print STDERR "ERROR unable to parse: $filename, $experiment\n";
				}
				
				
				# try parsing the experiment name, which 
				if (defined($experiment)) {
				
					$xparams{type} = "synth";
								$xparams{layout} = "separate";
								$xparams{sys} = "?";
								$xparams{nProcs} = "?";
								$xparams{nMaster} = 1;
								$xparams{nCompute} = "?";
								$xparams{nIO} = $4;
								$xparams{fileCount} = $1;
								$xparams{transport} = $2;
								$xparams{ioGroupSize} = $5;
								$xparams{bufferSize} = $3;
								$xparams{dataSize} = $6;
								$xparams{blocking} = $8;
								$xparams{compression} = $7;
								$xparams{ost} = "n";
				
				
					if (defined($expdates{$experiment})) {
						# real time. so now get the time from the file and mark the experiment
						$expFileTime = $expdates{$experiment};
						$expused{$experiment}++;
					}
					 
				}
				
			}
							# now try to find the terminal date.
			if (defined($experiment)) {
			
				$len = length($data) - $currPos;
				$endTime = checkForError(substr($data, $currPos, $len), $runEnd, $runEnd);
					
				$outputstr .= "$dn,$runname,$experiment,$startTime,";
				if (defined($endTime)) {
					$outputstr .= "$endTime";		
				} else {
					$outputstr .= "UNDEFINED";
				}
				if (defined($expFileTime)) {
					$outputstr .= ",$expFileTime";			
				} else {
					$outputstr .= ",NA,NA";
				}
		
				foreach my $colN (@colNames) {
					if (defined($fparams) && defined($fparams->{$colN})) {
						$outputstr .= ",$fparams->{$colN}";
					} else {
						$outputstr .= ",";
					}
					if (defined($xparams{$colN})) {
						$outputstr .= ",$xparams{$colN}";
					} else {
						$outputstr .= ",";
					}
				}
				$outputstr .= "\n";
			} else {
				if (defined($runStart)) {
					$outputstr .= "$dn,$runname,UNKNOWN,$runStart,$runEnd,NA\n";
				} else {
					$outputstr .= "$dn,$runname,UNKNOWN,UNKNOWN,$runEnd,NA\n";
					
				}
			}
				
			
			
		} else {
		
			while ( $data =~ /([^\n]*? E[DS]T 201[23])\n(=[^\n]*?=\n)?(mpirun|aprun|dir to create:) (.*? )([^ ]+?\.[np][0-9]+\.f[^\/ \n]+?)([\/, ].*?)?\n/g ) { 
				
				# if we've found experiment?
				if (defined($experiment)) {
					$len = pos($data) - $currPos;
					$endTime = checkForError(substr($data, $currPos, $len), reformatTimeString($1), $runEnd);
		
					# output previously found experiment
					$outputstr .= "$dn,$runname,$experiment,$startTime,";
					if (defined($endTime)) {
						$outputstr .= "$endTime";		
					} else {
						$outputstr .= "UNDEFINED";
					}
					if (defined($expFileTime)) {
						$outputstr .= ",$expFileTime";			
					} else {
						$outputstr .= ",NA";
					}
					
					# output the captured parameters
					foreach my $colN (@colNames) {
						if (defined($fparams) && defined($fparams->{$colN})) {
							$outputstr .= ",$fparams->{$colN}";
						} else {
							$outputstr .= ",";
						}
						if (defined($xparams{$colN})) {
							$outputstr .= ",$xparams{$colN}";
						} else {
							$outputstr .= ",";
						}
					}
					$outputstr .= "\n";
				} 
				
				$currPos = pos($data);
		
				# now set new experiment data
				$startTime = reformatTimeString($1);
				$endTime = undef;
				$experiment = $5;
				$fparams = undef;
				%xparams = ();
				$expFileTime = undef;
				$count++;
				
				my($marker) = $3;
				my($paramblock) = $6;
				my($executable) = $4;
				$executable =~ /([^\/ \.]+\.exe)/;
				$executable = $1;
				
				# try parsing the experiment name, which 
				if (defined($experiment)) {
					
					$fparams = parseFileName($experiment);
					if (!defined($fparams)) {
						print STDERR "ERROR unable to parse: $filename, $experiment\n";
					}
									
					if ($marker =~ /(mpi|ap)run/) {
						# then $paramblock has the parameters.  need to know the executable type
						if ($executable =~ /SynData_Full\.exe/) {
							if ($paramblock =~ /(\d+) [cg]pu ([^ ]+) (\d+) (\d+) \d+ ([\-]?\d+) \d+ (\d+) (\w+) (\w+)/ ) {
								$xparams{type} = "synth";
								$xparams{layout} = "separate";
								$xparams{sys} = "?";
								$xparams{nProcs} = "?";
								$xparams{nMaster} = 1;
								$xparams{nCompute} = "?";
								$xparams{nIO} = $4;
								$xparams{fileCount} = $1;
								$xparams{transport} = $2;
								$xparams{ioGroupSize} = $5;
								$xparams{bufferSize} = $3;
								$xparams{dataSize} = $6;
								$xparams{blocking} = $8;
								$xparams{compression} = $7;
								$xparams{ost} = "n";
								
							} else {
								print STDERR "1. $filename: $executable: $paramblock\n";
							}
						} elsif ($executable =~ /Process_test\.exe/) {
							if ($paramblock =~ /(\d+) [cg]pu ([^ ]+) (\d+) (\d+) \d+ ([\-]?\d+) \d+ (\w+) (\w+)/ ) {
								$xparams{type} = "TCGA";
								$xparams{layout} = "separate";
								$xparams{sys} = "?";
								$xparams{nProcs} = "?";
								$xparams{nMaster} = 1;
								$xparams{nCompute} = "?";
								$xparams{nIO} = $4;
								$xparams{fileCount} = $1;
								$xparams{transport} = $2;
								$xparams{ioGroupSize} = $5;
								$xparams{bufferSize} = $3;
								$xparams{dataSize} = 4096;
								$xparams{ost} = "n";
								$xparams{blocking} = $7;
								$xparams{compression} = $6;
							} elsif ($paramblock =~ /(\d+) [cg]pu ([^ ]+) (\d+) (\d+) \d+ ([\-]?\d+) \d+ (\d+)/ ) {
								$xparams{type} = "TCGA";
								$xparams{layout} = "separate";
								$xparams{sys} = "?";
								$xparams{nProcs} = "?";
								$xparams{nMaster} = 1;
								$xparams{nCompute} = "?";
								$xparams{nIO} = $4;
								$xparams{fileCount} = $1;
								$xparams{transport} = $2;
								$xparams{ioGroupSize} = $5;
								$xparams{bufferSize} = $3;
								$xparams{dataSize} = 4096;
								$xparams{ost} = "n";
								$xparams{blocking} = $6;
								$xparams{compression} = "n";
								
							} else {
								print STDERR "2. $filename: $executable: $paramblock\n";
							}
	
						} elsif ($executable =~ /nu-segment-scio\.exe/) {
							if ($paramblock =~ /(\d+) ([^ ]+) [cg]pu/ ) {
								$xparams{type} = "TCGA";
								$xparams{layout} = "baseline";
								$xparams{sys} = "?";
								$xparams{nProcs} = "?";
								$xparams{nMaster} = 1;
								$xparams{nCompute} = "all";
								$xparams{nIO} = "all";
								$xparams{fileCount} = $1;
								$xparams{transport} = "opencv";
								$xparams{ioGroupSize} = 1;
								$xparams{bufferSize} = 1;
								$xparams{dataSize} = 4096;
								$xparams{blocking} = "n";
								$xparams{compression} = "n";
								$xparams{ost} = "n";
								
							} else {
								print STDERR "3. $filename: $executable: $paramblock\n";
							}
							
						} elsif ($executable =~ /nu-segment-scio-adios\.exe/) {
							if ($paramblock =~ /([^ ]+) (\d+) (\d+) [cg]pu ([\-]?\d+) \d+/ ) {
								$xparams{type} = "TCGA";
								$xparams{layout} = "coloc";
								$xparams{sys} = "?";
								$xparams{nProcs} = "?";
								$xparams{nMaster} = 1;
								$xparams{nCompute} = "all";
								$xparams{nIO} = "all";
								$xparams{fileCount} = $2;
								$xparams{transport} = $1;
								$xparams{ioGroupSize} = $4;
								$xparams{bufferSize} = $3;
								$xparams{dataSize} = 4096;
								$xparams{blocking} = "n";
								$xparams{compression} = "n";
								$xparams{ost} = "n";
								
							} else {
								print STDERR "4. $filename: $executable: $paramblock\n";
							}
						} else {
							print STDERR "5. $filename: $executable: $paramblock\n";
						}
					}
					
					
					if (defined($expdates{$experiment})) {
						# real time. so now get the time from the file and mark the experiment
						$expFileTime = $expdates{$experiment};
						$expused{$experiment}++;
					} 
					
				}
				
			}
			
			# now try to find the terminal date.
			if (defined($experiment)) {
			
				$len = length($data) - $currPos;
				$endTime = checkForError(substr($data, $currPos, $len), $runEnd, $runEnd);
					
				$outputstr .= "$dn,$runname,$experiment,$startTime,";
				if (defined($endTime)) {
					$outputstr .= "$endTime";		
				} else {
					$outputstr .= "UNDEFINED";
				}
				if (defined($expFileTime)) {
					$outputstr .= ",$expFileTime";			
				} else {
					$outputstr .= ",NA,NA";
				}
		
				foreach my $colN (@colNames) {
					if (defined($fparams) && defined($fparams->{$colN})) {
						$outputstr .= ",$fparams->{$colN}";
					} else {
						$outputstr .= ",";
					}
					if (defined($xparams{$colN})) {
						$outputstr .= ",$xparams{$colN}";
					} else {
						$outputstr .= ",";
					}
				}
				$outputstr .= "\n";
			} else {
				if (defined($runStart)) {
					$outputstr .= "$dn,$runname,UNKNOWN,$runStart,$runEnd,NA\n";
				} else {
					$outputstr .= "$dn,$runname,UNKNOWN,UNKNOWN,$runEnd,NA\n";
					
				}
			}
			
	#	print "$filename : $count\n";
	
		}
	}

	# now output the ones that have not been marked
	foreach my $key (keys %expused) {
		if ($expused{$key} == 0) {
			$outputstr .= "$dn,UNKNOWN,$key,UNKNOWN,UNKNOWN,$expdates{$key}";

			my($xfn) = $dirname . "/" . $key;
			
			my($fparams) = undef;
			$fparams = parseFileName($xfn, "1", "1") if ($key !~ /^na-POSIX$/);
			if (defined($fparams)) {
				$outputstr .= ",$fparams->{$_}," foreach (@colNames);
				$outputstr .= "\n";
			} else {
				print STDERR "ERROR unable to parse 2: $dirname, $xfn\n";
			}
		}
	}

	$pm->finish(0, \$outputstr);
}

$pm->wait_all_children;
